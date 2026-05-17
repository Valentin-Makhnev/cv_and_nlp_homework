import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


class Config:
    model_name = 'bert-base-uncased'
    max_len = 128
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 5
    warmup_ratio = 0.1
    weight_decay = 0.01
    dropout = 0.1
    num_classes = 4
    
    model_save_path = "best_bert_model.pth"

config = Config()


class AGNewsBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return input_ids, attention_mask, labels


class BERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ====================== Обучение ======================
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Обучение"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Оценка"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_f1s, val_f1s):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Обучающие потери')
    axes[0].plot(val_losses, label='Валидационные потери')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Потери')
    axes[0].set_title('Динамика потерь при обучении и валидации')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_f1s, label='Train F1')
    axes[1].plot(val_f1s, label='Val F1')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Динамика F1-меры при обучении и валидации')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("График обучения сохранен как training_history.png")

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок на тестовом наборе AG News')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Матрица ошибок сохранена как confusion_matrix.png")

def main():
    parser = argparse.ArgumentParser(description='Классификация текстов AG News')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'distilbert-base-uncased'],
                        help='Модель BERT для использования')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    args = parser.parse_args()
    
    config.model_name = args.model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    
    print("=" * 60)
    print(f"Классификация AG News с использованием {config.model_name}")
    print("=" * 60)
    
    print("\n1. Загрузка датасета...")
    dataset = load_dataset('ag_news')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    print(f"Размер обучающей выборки: {len(train_texts)}, Размер тестовой выборки: {len(test_texts)}")
    
    print("\n2. Инициализация токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("\n3. Создание датасетов...")
    train_dataset = AGNewsBERTDataset(train_texts, train_labels, tokenizer, config.max_len)
    test_dataset = AGNewsBERTDataset(test_texts, test_labels, tokenizer, config.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                             shuffle=False, collate_fn=collate_fn)
    
    print("\n4. Создание модели...")
    model = BERTClassifier(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, 
                      weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n5. Начало обучения...")
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nЭпоха {epoch+1}/{config.num_epochs}")
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_loss, val_f1, _, _ = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        print(f"Обучающие потери: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Валидационные потери: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Сохранена лучшая модель с F1: {best_val_f1:.4f}")
    
    print("\n6. Финальная оценка...")
    model.load_state_dict(torch.load(config.model_save_path))
    _, final_f1, all_preds, all_labels = evaluate(model, test_loader, criterion)
    
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")
    print(f"Тестовый Macro F1: {final_f1:.4f}")
    
    if final_f1 >= 0.95:
        print(f"\nЦелевой показатель F1 >= 0.95 достигнут!")
    else:
        print(f"\nЦелевой показатель не достигнут. Получено {final_f1:.4f}, требуется >= 0.95")
    
    print("\n7. Создание визуализаций...")
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s)
    plot_confusion_matrix(all_labels, all_preds, ['Мир', 'Спорт', 'Бизнес', 'Наука/Технологии'])
    
    print("\n8. Отчет по классификации:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Мир', 'Спорт', 'Бизнес', 'Наука/Технологии']))
    
    print("\n9. Примеры предсказаний:")
    examples = [
        "Apple представила новый iPhone с передовыми AI функциями",
        "Манчестер Юнайтед выиграл титул Премьер-лиги после драматичного финального матча",
        "Саммит по изменению климата завершился новыми целями по сокращению выбросов углерода",
        "Фондовые рынки растут на фоне сигналов ФРС о снижении ставок"
    ]
    
    model.eval()
    class_names = ['Мир', 'Спорт', 'Бизнес', 'Наука/Технологии']
    for ex in examples:
        encoding = tokenizer(ex, truncation=True, padding='max_length', 
                           max_length=config.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            pred = torch.argmax(output, dim=1).item()
        
        print(f"Текст: {ex[:50]}... -> {class_names[pred]}")
    
    print(f"\n{'='*60}")
    print(f"Модель сохранена как: {config.model_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()