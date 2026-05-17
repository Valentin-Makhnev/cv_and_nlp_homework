import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42)

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        input_dim = latent_dim + num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        batch_size = z.shape[0]
        one_hot_labels = F.one_hot(labels, num_classes=10).float()
        combined = torch.cat([z, one_hot_labels], dim=1)
        output = self.fc(combined)
        output = output.view(batch_size, 1, 28, 28)
        return output

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        input_dim = 28*28 + num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        batch_size = img.shape[0]
        img_flat = img.view(batch_size, -1)
        one_hot_labels = F.one_hot(labels, num_classes=10).float()
        combined = torch.cat([img_flat, one_hot_labels], dim=1)
        output = self.fc(combined)
        return output

class ConditionalGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.generator = ConditionalGenerator(latent_dim)
        self.discriminator = ConditionalDiscriminator()
        
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        
        self.validation_labels = torch.arange(10).repeat(6)
        self.validation_z = torch.randn(60, latent_dim)
    
    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real_imgs, labels = batch
        batch_size = real_imgs.shape[0]
        
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Обучение дискриминатора
        real_validity = self.discriminator(real_imgs, labels)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z, labels)
        fake_validity = self.discriminator(fake_imgs.detach(), labels)
        d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        wrong_labels = torch.randint(0, 10, (batch_size,)).to(self.device)
        wrong_validity = self.discriminator(real_imgs, wrong_labels)
        d_wrong_loss = self.adversarial_loss(wrong_validity, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss + d_wrong_loss
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Обучение генератора
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z, labels)
        validity = self.discriminator(fake_imgs, labels)
        g_loss = self.adversarial_loss(validity, real_labels)
        
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)
        
        return {'d_loss': d_loss, 'g_loss': g_loss}
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        save_dir = 'generated_images'
        os.makedirs(save_dir, exist_ok=True)
        
        z = self.validation_z.type_as(next(self.generator.parameters()))
        labels = self.validation_labels.type_as(z).long()
        sample_imgs = self.generator(z, labels)
        
        fig, axes = plt.subplots(6, 10, figsize=(15, 9))
        for i in range(60):
            ax = axes[i // 10, i % 10]
            img = sample_imgs[i].squeeze().detach().cpu().numpy()
            img = (img + 1) / 2.0
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{labels[i].item()}', fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'conditional_gan_epoch_{self.current_epoch:03d}.png'))
        plt.close()

# Главная функция
if __name__ == '__main__':
    print("Загрузка данных MNIST...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Датасет загружен. Размер: {len(train_dataset)} изображений")
    
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = images[i].squeeze().numpy()
        img = (img + 1) / 2.0
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("Создание модели...")
    model = ConditionalGAN(latent_dim=100, lr=0.0002)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        mode='min',
        save_top_k=1,
        filename='conditional-gan-mnist-{epoch:02d}-{g_loss:.4f}'
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )
    
    print("Начало обучения...")
    trainer.fit(model, train_loader)
    
    print("Обучение завершено. Генерация результатов...")
    
    model.eval()
    with torch.no_grad():
        labels_to_generate = torch.arange(10).repeat(6).type_as(next(model.generator.parameters())).long()
        z = torch.randn(60, 100)
        z = z.type_as(next(model.generator.parameters()))
        fake_imgs = model.generator(z, labels_to_generate)
    
    fig, axes = plt.subplots(6, 10, figsize=(15, 9))
    for i in range(60):
        ax = axes[i // 10, i % 10]
        img = fake_imgs[i].squeeze().cpu().numpy()
        img = (img + 1) / 2.0
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels_to_generate[i].item()}', fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Генерация конкретной цифры
    target_digit = 7
    num_samples = 10
    
    with torch.no_grad():
        z = torch.randn(num_samples, 100)
        z = z.type_as(next(model.generator.parameters()))
        labels = torch.full((num_samples,), target_digit).type_as(z).long()
        generated = model.generator(z, labels)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(num_samples):
        ax = axes[i // 5, i % 5]
        img = generated[i].squeeze().cpu().numpy()
        img = (img + 1) / 2.0
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Generated {target_digit}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()