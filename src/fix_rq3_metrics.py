# ============================================================
# 一次性修复脚本：重算 RQ3 病例级指标
# 原因：原代码对已 sigmoid 的概率再次 sigmoid（calculate_metrics
#       期望输入 logits），导致所有像素都被判为阳性，
#       rq3_cases.csv 的 dice 列无效。
# 做法：加载已保存的 rq3_attunet_dropout.pth（无需重新训练），
#       用正确的二值化重算每个测试病例的 Dice，
#       覆盖写回 rq3_cases.csv。
# ============================================================
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DATA_DIR = os.environ.get("BUSI_DATA_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Dataset_BUSI_with_GT"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 与主脚本一致的数据构建 ----------
def build_manifest(data_dir):
    manifest = []
    for category in ["benign", "malignant", "normal"]:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue
        for img_path in glob(os.path.join(category_dir, "*.png")):
            basename = os.path.basename(img_path)
            if "_mask" in basename:
                continue
            mask_paths = glob(os.path.join(category_dir, basename.replace(".png", "_mask*.png")))
            manifest.append({"image_path": img_path, "mask_paths": mask_paths,
                             "label": category, "has_mask": len(mask_paths) > 0,
                             "num_masks": len(mask_paths)})
    df = pd.DataFrame(manifest)
    return df[(df["label"].isin(["benign", "malignant"])) & (df["has_mask"])].reset_index(drop=True)

class BUSIDataset(Dataset):
    def __init__(self, df, img_size=256):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for m_path in row["mask_paths"]:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m)
        mask = (mask > 127).astype(np.float32)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

# ---------- 与主脚本一致的模型结构 ----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                 nn.BatchNorm2d(inter_channels))
        self.W_g = nn.Sequential(nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
                                 nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        x1, g1 = self.W_x(x), self.W_g(g)
        if x1.shape[2:] != g1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        return x * self.psi(self.relu(x1 + g1))

class UNet(nn.Module):
    def __init__(self, n_classes=1, dropout_p=0.0, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DoubleConv(1, 64); self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256); self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024, dropout_p=dropout_p)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        if use_attention: self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512, dropout_p=dropout_p)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        if use_attention: self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256, dropout_p=dropout_p)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        if use_attention: self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128, dropout_p=dropout_p)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        if use_attention: self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64, dropout_p=dropout_p)
        self.out = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([self.att4(e4, d4) if self.use_attention else e4, d4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([self.att3(e3, d3) if self.use_attention else e3, d3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([self.att2(e2, d2) if self.use_attention else e2, d2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([self.att1(e1, d1) if self.use_attention else e1, d1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()

# ---------- 复现与主脚本相同的数据划分 ----------
manifest_df = build_manifest(DATA_DIR)
train_val_df, test_df = train_test_split(manifest_df, test_size=0.2,
                                         stratify=manifest_df["label"], random_state=42)
print(f"test cases: {len(test_df)}")

mc_test_ds = BUSIDataset(test_df)
mc_test_loader = DataLoader(mc_test_ds, batch_size=16, shuffle=False, num_workers=0)

# ---------- 加载已训练的模型 ----------
mc_model = UNet(use_attention=True, dropout_p=0.3).to(device)
mc_model.load_state_dict(torch.load("rq3_attunet_dropout.pth", map_location=device))
print("loaded rq3_attunet_dropout.pth")

# ---------- 1) eval 模式的确定性 Test Dice（报告用） ----------
mc_model.eval()
dices = []
with torch.no_grad():
    for images, masks in mc_test_loader:
        images, masks = images.to(device), masks.to(device)
        preds_bin = (torch.sigmoid(mc_model(images)) > 0.5).float()
        inter = (preds_bin * masks).sum(dim=[1, 2, 3])
        union = preds_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
        dices.extend(((2 * inter + 1e-6) / (union + 1e-6)).cpu().numpy())
print(f"[deterministic eval] test Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")

# ---------- 2) MC Dropout：正确二值化重算病例级 Dice ----------
enable_dropout_only(mc_model)
case_uncertainties, case_unc_global, case_dices = [], [], []
with torch.no_grad():
    for images, masks in tqdm(mc_test_loader, desc="MC Dropout re-eval"):
        images, masks = images.to(device), masks.to(device)
        preds = torch.stack([torch.sigmoid(mc_model(images)) for _ in range(20)])

        batch_var = preds.var(dim=0)

        # 正确做法：先对 20 次概率取均值，再用 0.5 阈值二值化（不再过 sigmoid）
        mean_probs = preds.mean(dim=0)
        preds_bin = (mean_probs > 0.5).float()
        inter = (preds_bin * masks).sum(dim=[1, 2, 3])
        union = preds_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
        batch_dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu().numpy()

        # 病灶区域聚焦的不确定性：只在预测为病灶的像素上平均方差
        # （全图平均会被大片自信背景稀释，信号消失；病灶区域才是医生关心的）
        for idx in range(images.size(0)):
            fg = preds_bin[idx, 0] > 0
            if fg.any():
                u_fg = batch_var[idx, 0][fg].mean().item()
            else:
                u_fg = batch_var[idx, 0].mean().item()  # 没有预测前景时退回全图平均
            case_uncertainties.append(u_fg)
            case_unc_global.append(batch_var[idx].mean().item())
            case_dices.append(batch_dice[idx])

pd.DataFrame({"uncertainty": case_uncertainties,
              "uncertainty_global": case_unc_global,
              "dice": case_dices}).to_csv("rq3_cases.csv", index=False)
print("rq3_cases.csv 已用正确指标重写（uncertainty=病灶区域聚焦, uncertainty_global=全图平均）")

corr, p_val = spearmanr(case_uncertainties, case_dices)
corr_g, p_val_g = spearmanr(case_unc_global, case_dices)
is_failed = [1 if d < 0.5 else 0 for d in case_dices]
print(f"cases: {len(case_dices)}, failed(Dice<0.5): {sum(is_failed)}, mean Dice: {np.mean(case_dices):.4f}")
print(f"Spearman (foreground): {corr:.4f} (p={p_val:.4e})")
print(f"Spearman (global avg): {corr_g:.4f} (p={p_val_g:.4e})")
if 0 < sum(is_failed) < len(is_failed):
    print(f"AUROC (foreground): {roc_auc_score(is_failed, case_uncertainties):.4f}")
