import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms, models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class FreiHand3DDataset(Dataset):
    def __init__(self, data_dir, xyz_normalized, xyz_original, indices=None, img_size=224, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.indices = indices if indices is not None else list(range(len(xyz_normalized)))
        self.xyz_normalized = xyz_normalized
        self.xyz_original = xyz_original
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = os.path.join(self.data_dir, "evaluation", "rgb", f"{real_idx:08d}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        xyz_norm = torch.tensor(self.xyz_normalized[real_idx], dtype=torch.float32).flatten()
        xyz_orig = torch.tensor(self.xyz_original[real_idx], dtype=torch.float32)
        return image, xyz_norm, xyz_orig


class Keypoint3DRegressor(nn.Module):
    def __init__(self, num_keypoints=21, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints * 3)
        )
    
    def forward(self, x):
        return self.backbone(x)

class Keypoint3DModule(pl.LightningModule):
    def __init__(self, num_keypoints=21, learning_rate=1e-3, xyz_scale=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = Keypoint3DRegressor(num_keypoints=num_keypoints)
        self.num_keypoints = num_keypoints
        self.xyz_scale = xyz_scale
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def compute_mpjpe_3d(self, pred, target):
        pred = pred.view(-1, self.num_keypoints, 3) * self.xyz_scale
        target = target.view(-1, self.num_keypoints, 3)
        distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
        return distances.mean() * 1000
    
    def training_step(self, batch, batch_idx):
        images, xyz_norm, xyz_orig = batch
        pred = self.forward(images)
        loss = self.criterion(pred, xyz_norm)
        mpjpe = self.compute_mpjpe_3d(pred, xyz_orig)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mpjpe', mpjpe, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, xyz_norm, xyz_orig = batch
        pred = self.forward(images)
        loss = self.criterion(pred, xyz_norm)
        mpjpe = self.compute_mpjpe_3d(pred, xyz_orig)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mpjpe', mpjpe, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, xyz_norm, xyz_orig = batch
        pred = self.forward(images)
        loss = self.criterion(pred, xyz_norm)
        mpjpe = self.compute_mpjpe_3d(pred, xyz_orig)
        self.log('test_loss', loss)
        self.log('test_mpjpe', mpjpe)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.model.backbone.conv1.parameters(), 'lr': 1e-5},
            {'params': self.model.backbone.bn1.parameters(), 'lr': 1e-5},
            {'params': self.model.backbone.layer1.parameters(), 'lr': 5e-5},
            {'params': self.model.backbone.layer2.parameters(), 'lr': 1e-4},
            {'params': self.model.backbone.layer3.parameters(), 'lr': 2e-4},
            {'params': self.model.backbone.layer4.parameters(), 'lr': 5e-4},
            {'params': self.model.backbone.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}
        }

# Обучение

def prepare_data(data_dir="./freihand_data"):
    with open(os.path.join(data_dir, "evaluation_xyz.json"), "r") as f:
        xyz = np.array(json.load(f))
    xyz_root = xyz - xyz[:, 0:1, :]
    xyz_norm = xyz_root / 0.2
    return xyz, xyz_root, xyz_norm, 0.2

def train_model():
    IMG_SIZE = 224
    BATCH_SIZE = 32
    MAX_EPOCHS = 200
    LR = 1e-3
    
    xyz, xyz_root, xyz_norm, scale = prepare_data()
    indices = np.random.permutation(len(xyz))
    train_idx = indices[:int(0.7*len(xyz))].tolist()
    val_idx = indices[int(0.7*len(xyz)):int(0.85*len(xyz))].tolist()
    test_idx = indices[int(0.85*len(xyz)):].tolist()
    
    train_loader = DataLoader(
        FreiHand3DDataset("./freihand_data", xyz_norm, xyz_root, train_idx, IMG_SIZE, True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        FreiHand3DDataset("./freihand_data", xyz_norm, xyz_root, val_idx, IMG_SIZE, False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        FreiHand3DDataset("./freihand_data", xyz_norm, xyz_root, test_idx, IMG_SIZE, False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    
    model = Keypoint3DModule(learning_rate=LR, xyz_scale=scale)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(monitor='val_mpjpe', mode='min', save_top_k=1),
            EarlyStopping(monitor='val_mpjpe', patience=15, mode='min')
        ],
        precision='16-mixed' if torch.cuda.is_available() else '32'
    )
    trainer.fit(model, train_loader, val_loader)
    
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = Keypoint3DModule.load_from_checkpoint(best_path, num_keypoints=21, xyz_scale=scale)
    
    test_results = trainer.test(model, test_loader)
    test_mpjpe = test_results[0]['test_mpjpe']
    
    print(f"\nТестовый MPJPE: {test_mpjpe:.2f} mm")
    return test_mpjpe

if __name__ == "__main__":
    pl.seed_everything(123)
    final = train_model()
    print(f"\nИтог: {final:.2f} mm")
    if final <= 20.0:
        print("Цель достигнута")
    else:
        print(f"Осталось: {final - 20.0:.2f} mm")