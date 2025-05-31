import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Dict
import json

def preprocess_chinese_text(text):
    """
    预处理中文文本
    1. 去除文本中的多余空格
    2. 保留基本标点符号
    3. 去除特殊字符
    """
    if not isinstance(text, str):
        return ""
    
    # 去除文本中的所有空格
    text = re.sub(r'\s+', '', text)
    
    # 保留中文标点符号和基本英文标点
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65\u0000-\u007f]', '', text)
    
    return text

def load_data(data_dir, language='en'):
    """
    加载处理好的数据
    """
    # 根据语言选择正确的目录
    lang_dir = os.path.join(data_dir, 'processed', language)
    if not os.path.exists(lang_dir):
        raise ValueError(f"数据目录不存在: {lang_dir}")
    
    # 加载训练集
    train_file = os.path.join(lang_dir, 'train.json')
    if not os.path.exists(train_file):
        raise ValueError(f"训练数据文件不存在: {train_file}")
    train_df = pd.read_json(train_file)
    
    # 加载测试集
    test_file = os.path.join(lang_dir, 'test.json')
    if not os.path.exists(test_file):
        raise ValueError(f"测试数据文件不存在: {test_file}")
    test_df = pd.read_json(test_file)
    
    # 对中文文本进行预处理
    if language == 'zh':
        train_df['text'] = train_df['text'].apply(preprocess_chinese_text)
        test_df['text'] = test_df['text'].apply(preprocess_chinese_text)
    
    print(f"\n数据集统计:")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"训练集标签分布:\n{train_df['label'].value_counts()}")
    print(f"测试集标签分布:\n{test_df['label'].value_counts()}\n")
    
    return (
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        test_df['text'].tolist(),
        test_df['label'].tolist()
    )

def create_data_loaders(train_texts, train_labels, test_texts, test_labels, 
                       tokenizer, batch_size=16, max_length=512):
    """
    创建训练集和验证集的数据加载器
    
    Args:
        train_texts: 训练集文本
        train_labels: 训练集标签
        test_texts: 测试集文本
        test_labels: 测试集标签
        tokenizer: 分词器
        batch_size: 批量大小
        max_length: 最大序列长度
    """
    # 创建数据集
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader 

def load_multilingual_data(chinese_data_dir, english_data_dir):
    """
    加载中英文数据
    
    Args:
        chinese_data_dir: 中文数据目录
        english_data_dir: 英文数据目录
    
    Returns:
        texts: 合并后的文本列表
        labels: 合并后的标签列表
    """
    # 加载中文数据
    zh_texts = []
    zh_labels = []
    
    # 处理中文数据
    human_dir = os.path.join(chinese_data_dir, 'human')
    generated_dir = os.path.join(chinese_data_dir, 'generated')
    
    # 读取人类撰写的文本
    for filename in os.listdir(human_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(human_dir, filename)
            try:
                df = pd.read_json(file_path)
                # 预处理中文文本
                df['text'] = df['text'].apply(preprocess_chinese_text)
                zh_texts.extend(df['text'].tolist())
                zh_labels.extend([0] * len(df))  # 0表示人类撰写
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # 读取生成的文本
    for filename in os.listdir(generated_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(generated_dir, filename)
            try:
                df = pd.read_json(file_path)
                # 预处理中文文本
                df['text'] = df['text'].apply(preprocess_chinese_text)
                zh_texts.extend(df['text'].tolist())
                zh_labels.extend([1] * len(df))  # 1表示人工生成
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # 加载英文数据
    en_texts = []
    en_labels = []
    
    # 处理英文数据
    domains = ['wp', 'reuter', 'essay']
    for domain in domains:
        # 人类撰写的文本
        human_path = os.path.join(english_data_dir, domain, 'human')
        if os.path.exists(human_path):
            for root, _, files in os.walk(human_path):
                for file in files:
                    if file.endswith('.txt'):
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            en_texts.append(text)
                            en_labels.append(0)  # 0表示人类撰写
        
        # 生成的文本
        gpt_path = os.path.join(english_data_dir, domain, 'gpt')
        if os.path.exists(gpt_path):
            for root, _, files in os.walk(gpt_path):
                for file in files:
                    if file.endswith('.txt'):
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            en_texts.append(text)
                            en_labels.append(1)  # 1表示人工生成
    
    # 合并数据
    all_texts = zh_texts + en_texts
    all_labels = zh_labels + en_labels
    
    print("\n数据集统计:")
    print(f"中文数据数量: {len(zh_texts)}")
    print(f"英文数据数量: {len(en_texts)}")
    print(f"总数据量: {len(all_texts)}")
    print(f"标签分布: 人类撰写: {all_labels.count(0)}, 人工生成: {all_labels.count(1)}\n")
    
    # 打印一些示例
    print("\n处理后的中文文本示例:")
    for i in range(min(3, len(zh_texts))):
        print(f"示例 {i+1}: {zh_texts[i][:100]}...")
    
    return all_texts, all_labels 

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        } 

class GhostbusterDataProcessor:
    def __init__(self, data_dir: str):
        """初始化数据处理器
        Args:
            data_dir: ghostbuster-data数据集的根目录
        """
        self.data_dir = data_dir
        self.domains = ['reuter', 'essay', 'wp']
        
    def load_combined_features(self, file_path: str) -> np.ndarray:
        """加载combined特征文件
        Args:
            file_path: combined文件路径
        Returns:
            特征数组
        """
        features = []
        max_length = 0
        
        # 首先读取所有特征并找到最大长度
        with open(file_path, 'r') as f:
            for line in f:
                feature = list(map(float, line.strip().split()))
                features.append(feature)
                max_length = max(max_length, len(feature))
        
        # 将所有特征填充到相同长度
        padded_features = []
        for feature in features:
            if len(feature) < max_length:
                # 用0填充较短的特征向量
                padded_feature = feature + [0.0] * (max_length - len(feature))
                padded_features.append(padded_feature)
            else:
                padded_features.append(feature)
        
        return np.array(padded_features)
    
    def process_reuter_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """处理reuter领域的数据
        Returns:
            features: 合并后的特征矩阵
            labels: 对应的标签数组
        """
        features_list = []
        labels_list = []
        
        # 处理人类作者数据
        human_dir = os.path.join(self.data_dir, 'reuter', 'human')
        for author in os.listdir(human_dir):
            author_dir = os.path.join(human_dir, author)
            if not os.path.isdir(author_dir):
                continue
                
            # 读取ada和davinci的combined特征
            ada_path = os.path.join(author_dir, 'combined-ada.txt')
            davinci_path = os.path.join(author_dir, 'combined-davinci.txt')
            
            if os.path.exists(ada_path) and os.path.exists(davinci_path):
                ada_features = self.load_combined_features(ada_path)
                davinci_features = self.load_combined_features(davinci_path)
                
                # 确保两个特征矩阵有相同的样本数
                min_samples = min(len(ada_features), len(davinci_features))
                ada_features = ada_features[:min_samples]
                davinci_features = davinci_features[:min_samples]
                
                # 合并特征
                combined_features = np.concatenate([ada_features, davinci_features], axis=1)
                features_list.append(combined_features)
                labels_list.extend([0] * min_samples)  # 0表示人类作者
            
        # 处理AI生成数据
        for ai_type in ['claude', 'gpt']:
            ai_dir = os.path.join(self.data_dir, 'reuter', ai_type)
            if os.path.exists(ai_dir):
                for author in os.listdir(ai_dir):
                    author_dir = os.path.join(ai_dir, author)
                    if not os.path.isdir(author_dir):
                        continue
                    
                    ada_path = os.path.join(author_dir, 'combined-ada.txt')
                    davinci_path = os.path.join(author_dir, 'combined-davinci.txt')
                    
                    if os.path.exists(ada_path) and os.path.exists(davinci_path):
                        ada_features = self.load_combined_features(ada_path)
                        davinci_features = self.load_combined_features(davinci_path)
                        
                        # 确保两个特征矩阵有相同的样本数
                        min_samples = min(len(ada_features), len(davinci_features))
                        ada_features = ada_features[:min_samples]
                        davinci_features = davinci_features[:min_samples]
                        
                        # 合并特征
                        combined_features = np.concatenate([ada_features, davinci_features], axis=1)
                        features_list.append(combined_features)
                        labels_list.extend([1] * min_samples)  # 1表示AI生成
        
        if not features_list:
            return np.array([]), np.array([])
            
        # 确保所有特征矩阵有相同的维度
        feature_dims = [f.shape[1] for f in features_list]
        if len(set(feature_dims)) > 1:
            max_dim = max(feature_dims)
            padded_features = []
            for features in features_list:
                if features.shape[1] < max_dim:
                    padding = np.zeros((features.shape[0], max_dim - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                padded_features.append(features)
            features_list = padded_features
                
        return np.vstack(features_list), np.array(labels_list)
    
    def process_simple_domain(self, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        """处理essay和wp领域的数据
        Args:
            domain: 'essay'或'wp'
        Returns:
            features: 合并后的特征矩阵
            labels: 对应的标签数组
        """
        features_list = []
        labels_list = []
        
        # 处理人类数据
        human_dir = os.path.join(self.data_dir, domain, 'human')
        if os.path.exists(human_dir):
            ada_path = os.path.join(human_dir, 'combined-ada.txt')
            davinci_path = os.path.join(human_dir, 'combined-davinci.txt')
            
            if os.path.exists(ada_path) and os.path.exists(davinci_path):
                ada_features = self.load_combined_features(ada_path)
                davinci_features = self.load_combined_features(davinci_path)
                
                # 确保两个特征矩阵有相同的样本数
                min_samples = min(len(ada_features), len(davinci_features))
                ada_features = ada_features[:min_samples]
                davinci_features = davinci_features[:min_samples]
                
                # 合并特征
                combined_features = np.concatenate([ada_features, davinci_features], axis=1)
                features_list.append(combined_features)
                labels_list.extend([0] * min_samples)
        
        # 处理AI生成数据
        for ai_type in ['claude', 'gpt']:
            ai_dir = os.path.join(self.data_dir, domain, ai_type)
            if os.path.exists(ai_dir):
                ada_path = os.path.join(ai_dir, 'combined-ada.txt')
                davinci_path = os.path.join(ai_dir, 'combined-davinci.txt')
                
                if os.path.exists(ada_path) and os.path.exists(davinci_path):
                    ada_features = self.load_combined_features(ada_path)
                    davinci_features = self.load_combined_features(davinci_path)
                    
                    # 确保两个特征矩阵有相同的样本数
                    min_samples = min(len(ada_features), len(davinci_features))
                    ada_features = ada_features[:min_samples]
                    davinci_features = davinci_features[:min_samples]
                    
                    # 合并特征
                    combined_features = np.concatenate([ada_features, davinci_features], axis=1)
                    features_list.append(combined_features)
                    labels_list.extend([1] * min_samples)
        
        if not features_list:
            return np.array([]), np.array([])
            
        # 确保所有特征矩阵有相同的维度
        feature_dims = [f.shape[1] for f in features_list]
        if len(set(feature_dims)) > 1:
            max_dim = max(feature_dims)
            padded_features = []
            for features in features_list:
                if features.shape[1] < max_dim:
                    padding = np.zeros((features.shape[0], max_dim - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                padded_features.append(features)
            features_list = padded_features
                
        return np.vstack(features_list), np.array(labels_list)
    
    def process_all_data(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """处理所有领域的数据并划分训练集和测试集
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        Returns:
            包含所有处理后数据的字典
        """
        all_data = {}
        
        # 处理每个领域的数据
        for domain in self.domains:
            if domain == 'reuter':
                features, labels = self.process_reuter_data()
            else:
                features, labels = self.process_simple_domain(domain)
                
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
            
            all_data[domain] = {
                'train': {'features': X_train, 'labels': y_train},
                'test': {'features': X_test, 'labels': y_test}
            }
            
        return all_data 

def load_data(data_dir, domain):
    """加载指定领域的数据
    
    Args:
        data_dir: 数据根目录
        domain: 领域名称
    
    Returns:
        训练集、验证集和测试集的数据
    """
    domain_dir = os.path.join(data_dir, domain)
    if not os.path.exists(domain_dir):
        raise ValueError(f"领域数据目录不存在: {domain_dir}")
    
    # 加载训练集
    train_file = os.path.join(domain_dir, 'train.json')
    if not os.path.exists(train_file):
        raise ValueError(f"训练数据文件不存在: {train_file}")
    train_df = pd.read_json(train_file)
    
    # 加载测试集
    test_file = os.path.join(domain_dir, 'test.json')
    if not os.path.exists(test_file):
        raise ValueError(f"测试数据文件不存在: {test_file}")
    test_df = pd.read_json(test_file)
    
    # 从训练集中分割出验证集
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
    
    print(f"\n{domain}领域数据集统计:")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"训练集标签分布: 人类={len(train_df[train_df['label']==0])}, AI={len(train_df[train_df['label']==1])}")
    print(f"验证集标签分布: 人类={len(val_df[val_df['label']==0])}, AI={len(val_df[val_df['label']==1])}")
    print(f"测试集标签分布: 人类={len(test_df[test_df['label']==0])}, AI={len(test_df[test_df['label']==1])}")
    
    return train_df, val_df, test_df

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """创建数据加载器
    
    Args:
        train_dataset: 训练集数据
        val_dataset: 验证集数据
        test_dataset: 测试集数据
        batch_size: 批量大小
    
    Returns:
        训练集、验证集和测试集的数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader 