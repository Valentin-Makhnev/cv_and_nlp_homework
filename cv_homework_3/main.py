import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# ДАТАСЕТ
# ============================================================================

class CamVidDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=(384, 384), transform=None, cache_images=False):
        self.transform = transform
        self.img_size = img_size
        self.cache_images = cache_images
        self.samples = []
        self.cache = {} if cache_images else None
        
        txt_file = os.path.join(data_dir, split + '.txt')
        
        if not os.path.exists(txt_file):
            print(f"Файл {txt_file} не найден!")
            return
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_name = os.path.basename(parts[0])
                        mask_name = os.path.basename(parts[1])
                        img_path = os.path.join(data_dir, split, img_name)
                        mask_path = os.path.join(data_dir, split + 'annot', mask_name)
                        
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            self.samples.append((img_path, mask_path))
        
        print(f"Загружено {len(self.samples)} файлов из {split}")
        
        # Кэширование данных
        if cache_images:
            print(f"Кэширование {split} данных...")
            for idx in tqdm(range(len(self.samples)), desc="Caching"):
                self._load_and_cache(idx)
    
    def _load_and_cache(self, idx):
        """Загрузка и кэширование изображения"""
        if self.cache_images and idx in self.cache:
            return self.cache[idx]
        
        img_path, mask_path = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.img_size, Image.BILINEAR)
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(self.img_size, Image.NEAREST)
            
            image = np.array(image)
            mask = np.array(mask)
            mask = np.clip(mask, 0, 11)
            
            if self.cache_images:
                self.cache[idx] = (image, mask)
            
            return image, mask
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            dummy_image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            dummy_mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
            return dummy_image, dummy_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.cache_images and idx in self.cache:
            image, mask = self.cache[idx]
        else:
            image, mask = self._load_and_cache(idx)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].long()
        
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, torch.from_numpy(mask).long()

# ============================================================================
# АУГМЕНТАЦИИ
# ============================================================================

def get_train_transform(img_size=(384, 384)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size=(384, 384)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# МОДЕЛЬ
# ============================================================================

class ImprovedUNet(nn.Module):
    def __init__(self, n_classes=12):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=n_classes,
        )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

# ============================================================================
# COMBINED LOSS (CrossEntropy + Focal + Dice)
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=12).permute(0, 3, 1, 2).float()
        
        dice_loss = 0
        for c in range(12):
            if c == 11:
                continue
            pred_c = pred_softmax[:, c]
            target_c = target_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
        
        return dice_loss / 11

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=11):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        class_weights = torch.tensor([0.6, 1.2, 3.0, 0.7, 1.0, 1.0, 2.5, 2.0, 0.8, 3.0, 3.0, 0.0])
        self.class_weights = class_weights
        
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights.to(pred.device), 
                                   ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        dice_loss = self.dice_loss(pred, target)
        
        combined = focal_loss + self.alpha * dice_loss
        return combined

# ============================================================================
# МЕТРИКИ
# ============================================================================

def compute_iou_details(pred, target, num_classes=11):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).float().sum().item()
        union = (pred_cls | target_cls).float().sum().item()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(0.0)
    return np.mean(ious), ious

def validate(model, loader, device):
    model.eval()
    total_iou = 0
    all_class_ious = np.zeros(11)
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(images.size(0)):
                batch_iou, class_ious = compute_iou_details(preds[i].cpu(), masks[i].cpu())
                total_iou += batch_iou
                all_class_ious += class_ious
                num_batches += 1
    
    mean_iou = total_iou / num_batches if num_batches > 0 else 0
    mean_class_ious = all_class_ious / num_batches if num_batches > 0 else np.zeros(11)
    
    return mean_iou, mean_class_ious

# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def train_model(data_dir, img_size=(384, 384), batch_size=8, max_epochs=80):
    print("="*60)
    print(f"Device: {device}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print("="*60)
    
    if device.type == 'cpu':
        cache_images = True
        batch_size = min(batch_size, 6)
        img_size = (320, 320)
        print(f"Кэширование данных: ВКЛЮЧЕНО")
        print(f"Размер изображений: {img_size}")
        print(f"Batch size: {batch_size}")
    else:
        cache_images = False
    
    print("="*60)
    
    print("\nЗагрузка тренировочных данных...")
    train_dataset = CamVidDataset(data_dir, 'train', img_size, 
                                  get_train_transform(img_size), 
                                  cache_images=cache_images)
    
    print(f"\nЗагрузка валидационных данных...")
    val_dataset = CamVidDataset(data_dir, 'val', img_size, 
                                get_val_transform(img_size),
                                cache_images=cache_images)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Нет данных для обучения!")
        return None, None, 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    print("\nСоздание модели UNet с энкодером ResNet50...")
    model = ImprovedUNet(n_classes=12).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")
    
    criterion = CombinedLoss(alpha=0.3, gamma=2.0, ignore_index=11)
    
    optimizer = torch.optim.AdamW([
        {'params': model.model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.model.decoder.parameters(), 'lr': 5e-3},
        {'params': model.model.segmentation_head.parameters(), 'lr': 5e-3},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_iou = 0
    history = {'train_loss': [], 'val_iou': [], 'class_ious': []}
    patience_counter = 0
    
    print("\nНачало обучения...")
    print("="*60)
    
    for epoch in range(max_epochs):
        print(f"\nЭпоха {epoch+1}/{max_epochs}")
        
        try:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_iou, class_ious = validate(model, val_loader, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_iou'].append(val_iou)
            history['class_ious'].append(class_ious)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val mIoU: {val_iou:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            if (epoch + 1) % 10 == 0:
                class_names = ['Sky', 'Bld', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                              'Sign', 'Fence', 'Car', 'Ped', 'Bike']
                worst_classes = sorted(zip(class_names, class_ious), key=lambda x: x[1])[:3]
                print(f"Худшие классы: {worst_classes}")
            
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_iou': val_iou,
                    'class_ious': class_ious,
                }, 'best_unet_model.pth')
                print(f"Сохранена лучшая модель! (mIoU: {best_val_iou:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break
            
            if len(history['val_iou']) > 10:
                last_5 = history['val_iou'][-5:]
                if max(last_5) - min(last_5) < 0.003:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    print(f"Снижен LR до {optimizer.param_groups[0]['lr']:.6f}")
            
            if best_val_iou >= 0.7:
                print(f"\nЦель на эпохе {epoch+1}")
                break
                
        except Exception as e:
            print(f"Ошибка на эпохе {epoch+1}: {e}")
            continue
        
        gc.collect()
    
    print("\n" + "="*60)
    print(f"Обучение завершено! Лучший Val mIoU: {best_val_iou:.4f}")
    if best_val_iou >= 0.7:
        print("mIoU >= 0.7")
    else:
        print(f"Не хватает: {0.7 - best_val_iou:.4f}")
    print("="*60)
    
    return model, history, best_val_iou

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    DATA_DIR = './CamVid'
    
    if not os.path.exists(DATA_DIR):
        DATA_DIR = input("Укажите путь к датасету CamVid: ")
    
    if not os.path.exists(DATA_DIR):
        print(f"Папка {DATA_DIR} не найдена!")
        exit(1)
    
    model, history, best_iou = train_model(
        data_dir=DATA_DIR,
        img_size=(384, 384),
        batch_size=6,
        max_epochs=80
    )
    
    if history and len(history['val_iou']) > 0:
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_iou'], label='Val mIoU', color='green', linewidth=2)
        plt.axhline(y=0.7, color='red', linestyle='--', label='Target (0.7)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        plt.title(f'Mean IoU (Best: {best_iou:.4f})')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        if history['class_ious']:
            final_ious = history['class_ious'][-1]
            class_names = ['Sky', 'Bld', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                          'Sign', 'Fence', 'Car', 'Ped', 'Bike']
            colors = ['g' if iou > 0.7 else 'r' for iou in final_ious]
            plt.bar(range(11), final_ious, color=colors)
            plt.axhline(y=0.7, color='blue', linestyle='--', label='Target')
            plt.xlabel('Class')
            plt.ylabel('IoU')
            plt.title('IoU per Class')
            plt.xticks(range(11), class_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
    
    print("\n" + "="*60)
    print("Итоги выполнения задания")
    print("="*60)
    print("Использован предобученный UNet (segmentation_models_pytorch)")
    print("Аугментации из albumentations")
    print("OneCycleLR/CosineAnnealing scheduler")
    print("Combined Loss (Focal + Dice) для несбалансированных классов")
    print("AdamW с weight_decay и разными LR для разных частей")
    print(f"Целевая метрика mIoU: {best_iou:.4f}")
    print("="*60)

"""
Вывод:
Более мощный энкодер - ResNet50 имеет больше слоев и каналов, лучше извлекает сложные признаки (формы автомобилей, текстуры деревьев)
Combined loss (Focal + Dice): 
    Focal loss решает проблему дисбаланса классов (мало пикселей пешеходов/велосипедистов)
    Dice loss косвенно оптимизирует IoU метрику (имеет сильный сигнал на границах объектов, устойчив к дисбалансу)
Разные Learning Rates для разных частей - энкодер (1e-4) уже предобучен на ImageNet и не требует сильного изменения, декодер (5e-3) нужно обучить с нуля
Cosine Annealing scheduler - Позволяет "выпрыгивать" из локальных минимумов, периодически увеличивая learning rate
Усиленные аугментации - увеличивает разнообразие данных, модель учится инвариантности к освещению и ракурсам
Gradient clipping - предотвращает увеличение градиентов, делает обучение стабильнее
Больше эпох - больше эпох позволяет модели дообучиться до оптимума
Увеличение размера изображения - большие изображения сохраняют мелкие объекты (пешеходы, знаки)
"""