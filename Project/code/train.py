import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
import numpy as np
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertConfig
from models import TextDetectionModel, GhostbusterModel
from scipy.stats import entropy
import random

# 过滤掉Flash Attention相关的警告
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')

# 设置环境变量以使用传统attention实现
os.environ["TORCH_USE_CUDA_DSA"] = "0"

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def train_model(model, train_loader, val_loader, device, num_epochs=5, learning_rate=2e-5, output_dir=None, optimizer=None, scheduler=None):
    """训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        num_epochs: 训练轮数
        learning_rate: 学习率
        output_dir: 输出目录
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    
    Returns:
        训练好的模型
    """
    # 创建优化器和调度器（如果没有提供）
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    if scheduler is None:
        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
    
    # 创建损失函数
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # 创建梯度缩放器
    scaler = GradScaler()
    
    # 创建早停
    early_stopping = EarlyStopping(patience=3)
    
    # 记录最佳模型
    best_val_loss = float('inf')
    best_model_state = None
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            # 将数据移到GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用自动混合精度训练
            with autocast():
                outputs = model(**batch)
                loss = outputs['loss']
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            train_preds.extend(predictions.cpu().numpy())
            train_labels.extend(batch['labels'].cpu().numpy())
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        
        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with autocast():
                    outputs = model(**batch)
                    loss = outputs['loss']
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
    
        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # 打印训练信息
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, os.path.join(output_dir, 'best_model.pt'))
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, eval_loader, device):
    """评估模型
    
    Args:
        model: 要评估的模型
        eval_loader: 评估数据加载器
        device: 设备
    
    Returns:
        包含评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast():
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    # 计算评估指标
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }

def compute_metrics(preds, labels):
    """计算评估指标
    
    Args:
        preds: 预测标签列表
        labels: 真实标签列表
    
    Returns:
        包含各项指标的字典
    """
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    } 