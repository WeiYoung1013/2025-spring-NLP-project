import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class ResidualBlock(nn.Module):
    """简化的残差块"""
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return x + self.layer(x)

class TextDetectionModel(BertPreTrainedModel):
    """简化的基于BERT的文本检测模型"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        
        # 简化的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # 简化的分类器
        self.classifier = nn.Linear(256, 2)  # 二分类
        
        # 初始化权重
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        # 获取BERT输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        
        # 只使用[CLS]的输出
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # 特征提取
        features = self.feature_extractor(pooled_output)
        
        # 分类
        logits = self.classifier(features)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'features': features
        }

class SelfAttention(nn.Module):
    """自注意力层"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        weights = torch.softmax(self.attention(x), dim=1)
        return weights * x

class GhostbusterModel(nn.Module):
    """简化的特征检测模型"""
    def __init__(self, input_size):
        super().__init__()
        
        # 输入标准化
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # 简化的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def forward(self, features, labels=None):
        # 输入标准化
        features = self.input_bn(features)
        
        # 特征提取
        extracted_features = self.feature_extractor(features)
        
        # 分类
        logits = self.classifier(extracted_features)
        
        outputs = {
            'logits': logits,
            'features': extracted_features
        }
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs['loss'] = loss
        
        return outputs 