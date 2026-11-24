
# ===== Cell 0 =====
# 1. 挂载Google Drive (如果后续数据放在Drive里，这步很有用；如果只用sample_data，可以不执行)
# from google.colab import drive
# drive.mount('/content/drive')

# 2. 基础库导入与随机种子固定
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 注意：Windows 本地运行时 DataLoader 的 num_workers 已统一设为 0（多进程加载在 Windows 会崩溃）
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. 修改数据路径为你的实际路径（已改为本地 BUSI 路径）
DATA_DIR = os.environ.get("BUSI_DATA_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Dataset_BUSI_with_GT"))

# 无论从哪里启动脚本，输出文件（CSV/图）都统一写到本脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===== Cell 1 =====
# 4. 数据审计与构建 Manifest
def build_manifest(data_dir):
    manifest = []

    # 遍历三类文件夹
    for category in ['benign', 'malignant', 'normal']:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: {category_dir} not found!")
            continue

        # 获取该类别下所有图片
        image_paths = glob(os.path.join(category_dir, "*.png"))

        for img_path in image_paths:
            basename = os.path.basename(img_path)

            # 排除本身就是 mask 的文件
            if "_mask" in basename:
                continue

            # 查找对应的 mask 文件 (可能有一个或多个，例如 benign (1)_mask.png)
            # 假设 mask 命名规则为: 原图名 + "_mask.png" 或 "_mask_1.png"
            mask_pattern = basename.replace('.png', '_mask*.png')
            mask_paths = glob(os.path.join(category_dir, mask_pattern))

            manifest.append({
                'image_path': img_path,
                'mask_paths': mask_paths,
                'label': category,
                'has_mask': len(mask_paths) > 0,
                'num_masks': len(mask_paths)
            })

    df = pd.DataFrame(manifest)

    # 打印审计信息
    print(f"Total images found: {len(df)}")
    print("\nImage counts by category:")
    print(df['label'].value_counts())

    # 过滤出有mask的良性/恶性用于分割 (排除normal)
    df_seg = df[(df['label'].isin(['benign', 'malignant'])) & (df['has_mask'])].reset_index(drop=True)
    print(f"\n--- Segmentation Dataset (Benign + Malignant) ---")
    print(f"Total usable images: {len(df_seg)}")
    print(f"Images with multiple masks: {len(df_seg[df_seg['num_masks'] > 1])}")

    # 检查是否有缺失 mask 的异常情况
    missing_mask = df[(df['label'].isin(['benign', 'malignant'])) & (~df['has_mask'])]
    if len(missing_mask) > 0:
        print(f"\nWarning: Found {len(missing_mask)} benign/malignant images WITHOUT masks.")

    return df_seg

manifest_df = build_manifest(DATA_DIR)
manifest_df.head()

# ===== Cell 2 =====
# 本地替代 google.colab 的 InteractiveSheet：导出 CSV + 打印前几行
manifest_df.to_csv("manifest.csv", index=False)
print(manifest_df.head(10))
print("manifest 行数:", len(manifest_df))

# ===== Cell 3 =====
# ==================== Cell 3: Dataset ====================
class BUSIDataset(Dataset):
    def __init__(self, df, img_size=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment # 对应提案中的 augmentation on/off

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_paths = row['mask_paths']

        # 读图 (灰度)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # 读Mask并合并 (Union)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for m_path in mask_paths:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m) # 多mask取并集

        mask = (mask > 127).astype(np.float32) # 二值化

        # 归一化
        image = image.astype(np.float32) / 255.0

        # 数据增强 (只针对训练集)
        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # 转换为 Tensor -> (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# 测试 Dataset
test_ds = BUSIDataset(manifest_df, augment=False)
img, msk = test_ds[0]
print(f"Image shape: {img.shape}, Mask shape: {msk.shape}")


# ===== Cell 4 =====
# 7. 基础卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# 8. Attention Gate (Oktay et al., 2018)
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip connection feature, g: gating feature from decoder
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        # 如果尺寸不匹配，对gating进行上采样
        if x1.shape[2:] != g1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(x1 + g1)
        psi = self.psi(psi)
        return x * psi

# 9. U-Net 模型
class UNet(nn.Module):
    def __init__(self, n_classes=1, dropout_p=0.0, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout_p=dropout_p)

        # Decoder (包含Dropout用于MC Dropout)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        if use_attention:
            self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        if use_attention:
            self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        if use_attention:
            self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128, dropout_p=dropout_p)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        if use_attention:
            self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64, dropout_p=dropout_p)

        # Output
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if self.use_attention:
            e4_att = self.att4(e4, d4)
            d4 = torch.cat([e4_att, d4], dim=1)
        else:
            d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if self.use_attention:
            e3_att = self.att3(e3, d3)
            d3 = torch.cat([e3_att, d3], dim=1)
        else:
            d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if self.use_attention:
            e2_att = self.att2(e2, d2)
            d2 = torch.cat([e2_att, d2], dim=1)
        else:
            d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if self.use_attention:
            e1_att = self.att1(e1, d1)
            d1 = torch.cat([e1_att, d1], dim=1)
        else:
            d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# 实例化测试
model_unet = UNet(use_attention=False, dropout_p=0.3).to(device)
model_att_unet = UNet(use_attention=True, dropout_p=0.3).to(device)
print(f"U-Net params: {sum(p.numel() for p in model_unet.parameters())}")

# ===== Cell 5 =====
# 10. 损失函数与指标
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # preds: logits (B, 1, H, W)
        # targets: binary masks (B, 1, H, W)

        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Dice Loss (需要先经过sigmoid)
        preds_prob = torch.sigmoid(preds)
        intersection = (preds_prob * targets).sum()
        union = preds_prob.sum() + targets.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        return bce_loss + dice_loss

def calculate_metrics(preds, targets, threshold=0.5):
    # 转为二值
    preds_bin = (torch.sigmoid(preds) > threshold).float()

    # Flatten
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)

    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum()

    # 计算 Dice
    dice = (2. * intersection + 1e-6) / (union + 1e-6)

    # 计算 IoU
    iou = (intersection + 1e-6) / (preds_bin.sum() + targets.sum() - intersection + 1e-6)

    return dice.item(), iou.item()

# ===== Cell 6 =====
# 11. 训练与验证循环
import copy
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        dice, _ = calculate_metrics(outputs, masks)
        epoch_dice += dice

    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            dice, iou = calculate_metrics(outputs, masks)
            epoch_dice += dice
            epoch_iou += iou

    return epoch_loss / len(dataloader), epoch_dice / len(dataloader), epoch_iou / len(dataloader)

def fit_model(model, train_loader, val_loader, epochs, lr, device, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0.0
    early_stop_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # 早停与保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    return model, history

# ===== Cell 7 =====
# 12. RQ1: 5-fold CV
from sklearn.model_selection import StratifiedKFold

EPOCHS = 30   # 冒烟测试用 5；正式跑请改为 30~50（早停会自动收尾）
BATCH_SIZE = 16
LR = 1e-4

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 使用 label 做分层
X = manifest_df.index.values
y = manifest_df['label'].values

rq1_results = {'UNet': [], 'AttUNet': []}

print("========== 开始 RQ1: 5-Fold Cross Validation ==========")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/5 ---")

    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    val_df = manifest_df.iloc[val_idx].reset_index(drop=True)

    # DataLoader (训练集开启增强)
    train_ds = BUSIDataset(train_df, augment=True)
    val_ds = BUSIDataset(val_df, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 训练 Model A: U-Net ---
    print(">> Training Model A: U-Net")
    model_unet = UNet(use_attention=False, dropout_p=0.0).to(device) # RQ1不需要dropout
    trained_unet, _ = fit_model(model_unet, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)
    _, val_dice_unet, val_iou_unet = validate(trained_unet, val_loader, DiceBCELoss(), device)

    # --- 训练 Model B: Attention U-Net ---
    print(">> Training Model B: Attention U-Net")
    model_att = UNet(use_attention=True, dropout_p=0.0).to(device)
    trained_att, _ = fit_model(model_att, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)
    _, val_dice_att, val_iou_att = validate(trained_att, val_loader, DiceBCELoss(), device)

    rq1_results['UNet'].append({'dice': val_dice_unet, 'iou': val_iou_unet})
    rq1_results['AttUNet'].append({'dice': val_dice_att, 'iou': val_iou_att})

# 打印 RQ1 汇总结果
rq1_rows = [{'model': m, 'fold': i + 1, **r} for m, res in rq1_results.items() for i, r in enumerate(res)]
pd.DataFrame(rq1_rows).to_csv('rq1_results.csv', index=False)
print("RQ1 结果已保存: rq1_results.csv")

print("\n========== RQ1 Final Summary ==========")
for model_name, res in rq1_results.items():
    dices = [r['dice'] for r in res]
    ious = [r['iou'] for r in res]
    print(f"{model_name}: Dice = {np.mean(dices):.4f} ± {np.std(dices):.4f} | IoU = {np.mean(ious):.4f} ± {np.std(ious):.4f}")

# ===== Cell 8 =====
# 13. RQ2: Data Efficiency
from sklearn.model_selection import train_test_split

# 划分固定的 Train/Val/Test (60% / 20% / 20%)
train_val_df, test_df = train_test_split(manifest_df, test_size=0.2, stratify=manifest_df['label'], random_state=42)
train_df_full, val_df = train_test_split(train_val_df, test_size=0.25, stratify=train_val_df['label'], random_state=42) # 0.25 * 0.8 = 0.2

print(f"RQ2 Fixed Split: Train={len(train_df_full)}, Val={len(val_df)}, Test={len(test_df)}")

def stratified_prefix(df, frac, seed):
    """按类别分层打乱后取前 frac 部分 —— 同一 seed 下各 fraction 严格嵌套"""
    if frac >= 1.0:
        return df
    parts = []
    for lbl, g in df.groupby('label'):
        g = g.sample(frac=1.0, random_state=seed)
        parts.append(g.head(max(1, int(round(len(g) * frac)))))
    return pd.concat(parts)

fractions = [0.1, 0.25, 0.5, 1.0]
seeds = [42, 123, 2024]
augment_options = [False, True]
rq2_results = []

# 注意：为了节省时间，这里写的是单模型（Plain U-Net）的循环。
# 如果你需要 Attention U-Net，只需在内部加上同样的循环。
for aug in augment_options:
    for frac in fractions:
        for seed in seeds:
            print(f"\n--- RQ2: Frac={frac}, Aug={aug}, Seed={seed} ---")
            seed_everything(seed)

            # 采样训练集子集（嵌套：10% ⊂ 25% ⊂ 50% ⊂ 100%）
            train_df_sub = stratified_prefix(train_df_full, frac, seed)

            train_ds = BUSIDataset(train_df_sub, augment=aug)
            val_ds = BUSIDataset(val_df, augment=False)
            test_ds = BUSIDataset(test_df, augment=False)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            # 只跑 U-Net
            model = UNet(use_attention=False, dropout_p=0.0).to(device)
            # 调整 epochs，子集小的时候可以适当减少 epoch
            trained_model, _ = fit_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device, patience=15)

            # 在 Test set 上评估
            _, test_dice, test_iou = validate(trained_model, test_loader, DiceBCELoss(), device)

            rq2_results.append({
                'fraction': frac, 'augment': aug, 'seed': seed,
                'test_dice': test_dice, 'test_iou': test_iou
            })

# 展示 RQ2 结果表
df_rq2 = pd.DataFrame(rq2_results)
print("\n========== RQ2 Summary ==========")
print(df_rq2.groupby(['fraction', 'augment'])[['test_dice', 'test_iou']].agg(['mean', 'std']))
df_rq2.to_csv('rq2_results.csv', index=False)
print("RQ2 结果已保存: rq2_results.csv")

# ===== Cell 9 =====
# 14. RQ3: MC Dropout Uncertainty
import matplotlib.pyplot as plt

def enable_dropout_only(model):
    """只开启 Dropout 层，其余保持 eval —— 避免 BatchNorm 在 train 模式下引入额外噪声"""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()

def mc_dropout_predict(model, image_tensor, T=20):
    """
    输入单张图像(1, 1, 256, 256)，返回 (均值预测, 方差图)
    """
    enable_dropout_only(model) # 只开启 Dropout
    preds = []
    with torch.no_grad():
        for _ in range(T):
            pred = torch.sigmoid(model(image_tensor))
            preds.append(pred)

    preds = torch.stack(preds) # (T, 1, 1, 256, 256)
    mean_pred = preds.mean(dim=0).squeeze() # (256, 256)
    variance_map = preds.var(dim=0).squeeze() # (256, 256)
    return mean_pred.cpu().numpy(), variance_map.cpu().numpy()

# 假设我们取之前的某一个模型，并加上 dropout_p=0.3
# 为了演示，我们直接在这里用 test_df 训练一个带 Dropout 的模型
print("训练带 MC Dropout 的模型用于 RQ3...")
mc_train_ds = BUSIDataset(train_df_full, augment=True)
mc_val_ds = BUSIDataset(val_df, augment=False)
mc_test_ds = BUSIDataset(test_df, augment=False)

mc_train_loader = DataLoader(mc_train_ds, batch_size=BATCH_SIZE, shuffle=True)
mc_val_loader = DataLoader(mc_val_ds, batch_size=BATCH_SIZE, shuffle=False)
mc_test_loader = DataLoader(mc_test_ds, batch_size=BATCH_SIZE, shuffle=False)

mc_model = UNet(use_attention=True, dropout_p=0.3).to(device) # 用 Att-UNet + Dropout
mc_model, _ = fit_model(mc_model, mc_train_loader, mc_val_loader, epochs=EPOCHS, lr=LR, device=device)
torch.save(mc_model.state_dict(), "rq3_attunet_dropout.pth")
print("RQ3 模型权重已保存: rq3_attunet_dropout.pth")

# 对测试集进行 MC Dropout 评估
mc_model.eval() # 先切回 eval 模式测一次确定性指标（报告用）
_, mc_test_dice, mc_test_iou = validate(mc_model, mc_test_loader, DiceBCELoss(), device)
print(f"RQ3 模型（eval 模式）Test Dice: {mc_test_dice:.4f} | IoU: {mc_test_iou:.4f}")

# 这里快速跑几个样本看效果
print("\n生成不确定性图...")
num_samples = 4
fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

for i in range(num_samples):
    img_tensor, mask_tensor = mc_test_ds[i]
    img_input = img_tensor.unsqueeze(0).to(device)

    # MC Dropout 预测
    mean_pred, var_map = mc_dropout_predict(mc_model, img_input, T=20)

    # 画图
    axes[i, 0].imshow(img_tensor.squeeze(), cmap='gray'); axes[i, 0].set_title('Image'); axes[i, 0].axis('off')
    axes[i, 1].imshow(mask_tensor.squeeze(), cmap='gray'); axes[i, 1].set_title('GT Mask'); axes[i, 1].axis('off')
    axes[i, 2].imshow(mean_pred > 0.5, cmap='gray'); axes[i, 2].set_title('Pred Mask'); axes[i, 2].axis('off')
    im = axes[i, 3].imshow(var_map, cmap='hot'); axes[i, 3].set_title('Uncertainty Map'); axes[i, 3].axis('off')
    fig.colorbar(im, ax=axes[i, 3])

plt.tight_layout()
plt.savefig("rq3_uncertainty_grid.png", dpi=120)
print("RQ3 不确定性对比图已保存: rq3_uncertainty_grid.png")
plt.show()

# 真实场景下，你需要写一个循环计算所有 test case 的平均不确定性，并与 Dice 算 Spearman
# 以下是计算代码示例：

case_uncertainties = []
case_unc_global = []
case_dices = []

enable_dropout_only(mc_model) # 只开启 dropout（BatchNorm 保持 eval）
with torch.no_grad():
    for images, masks in tqdm(mc_test_loader, desc="Calculating RQ3 metrics"):
        images, masks = images.to(device), masks.to(device)
        preds = torch.stack([torch.sigmoid(mc_model(images)) for _ in range(20)])

        batch_var = preds.var(dim=0) # (B, 1, H, W)

        # 每个样本的 Dice：对 20 次概率取均值后直接以 0.5 二值化
        # （注意：不能再过 sigmoid —— calculate_metrics 期望输入 logits，
        #  而这里是概率，再过一次会让所有像素都超过阈值）
        mean_probs = preds.mean(dim=0)
        preds_bin = (mean_probs > 0.5).float()
        inter = (preds_bin * masks).sum(dim=[1, 2, 3])
        union = preds_bin.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3])
        batch_dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu().numpy()

        # 每个样本的不确定性：病灶区域聚焦（只在预测为病灶的像素上平均方差，
        # 避免全图平均被大片自信背景稀释）；同时保留全图平均作对照
        for idx in range(images.size(0)):
            fg = preds_bin[idx, 0] > 0
            if fg.any():
                u_fg = batch_var[idx, 0][fg].mean().item()
            else:
                u_fg = batch_var[idx, 0].mean().item()
            case_uncertainties.append(u_fg)
            case_unc_global.append(batch_var[idx].mean().item())
            case_dices.append(batch_dice[idx])

pd.DataFrame({'uncertainty': case_uncertainties, 'uncertainty_global': case_unc_global, 'dice': case_dices}).to_csv('rq3_cases.csv', index=False)
print("RQ3 病例级结果已保存: rq3_cases.csv")

# 1. Spearman correlation
corr, p_val = spearmanr(case_uncertainties, case_dices)
print(f"Spearman Correlation between Uncertainty and Dice: {corr:.4f} (p={p_val:.4e})")

# 2. AUROC of predicting failed cases (Dice < 0.5)
is_failed = [1 if d < 0.5 else 0 for d in case_dices]
if sum(is_failed) > 0 and sum(is_failed) < len(is_failed):
    auroc = roc_auc_score(is_failed, case_uncertainties)
    print(f"AUROC of Uncertainty score for predicting failed cases (Dice<0.5): {auroc:.4f}")
else:
    print("Not enough failed cases in this sample to calculate AUROC.")