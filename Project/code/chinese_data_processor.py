import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import numpy as np

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chinese_data_processing.log')
    ]
)

class ChineseTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_json_data(file_path):
    """加载JSON格式的数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_chinese_data(human_dir, generated_dir, domain):
    """处理中文数据集
    
    Args:
        human_dir: 人类写作文本目录
        generated_dir: AI生成文本目录
        domain: 领域名称 (news/webnovel/wiki)
    """
    logging.info(f"处理{domain}领域的数据...")
    
    # 加载人类写作数据
    if domain == 'webnovel':
        human_file = os.path.join(human_dir, f"{domain}.json")
    else:
        human_file = os.path.join(human_dir, f"{domain}-zh.json")
    
    human_data = load_json_data(human_file)
    human_texts = [item['output'] for item in human_data]  # 使用'output'字段作为文本
    human_labels = [0] * len(human_texts)  # 0表示人类写作
    
    # 加载AI生成数据
    if domain == 'webnovel':
        generated_file = os.path.join(generated_dir, f"{domain}.qwen2-72b-base.json")
    else:
        generated_file = os.path.join(generated_dir, f"{domain}-zh.qwen2-72b-base.json")
    
    logging.info(f"正在加载生成数据: {generated_file}")
    generated_data = load_json_data(generated_file)
    
    # 处理生成数据
    if isinstance(generated_data, dict):
        # 如果output是字典，获取其所有值
        if isinstance(generated_data['output'], dict):
            generated_texts = list(generated_data['output'].values())
        # 如果output是列表
        elif isinstance(generated_data['output'], list):
            generated_texts = generated_data['output']
        # 如果output是字符串
        else:
            generated_texts = [generated_data['output']]
    else:  # 列表格式
        generated_texts = [item['output'] for item in generated_data]
    
    # 过滤掉空文本
    generated_texts = [text for text in generated_texts if text and isinstance(text, str)]
    generated_labels = [1] * len(generated_texts)  # 1表示AI生成
    
    # 确保数据量平衡
    min_samples = min(len(human_texts), len(generated_texts))
    if min_samples < 100:
        raise ValueError(f"数据量太少（人类: {len(human_texts)}, AI: {len(generated_texts)}），至少需要100个样本")
    
    if len(human_texts) > min_samples:
        human_texts = human_texts[:min_samples]
        human_labels = human_labels[:min_samples]
    if len(generated_texts) > min_samples:
        generated_texts = generated_texts[:min_samples]
        generated_labels = generated_labels[:min_samples]
    
    # 合并数据
    all_texts = human_texts + generated_texts
    all_labels = human_labels + generated_labels
    
    # 创建DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    # 打印数据集统计信息
    logging.info(f"\n{domain}领域数据集统计：")
    logging.info(f"总样本数: {len(df)}")
    logging.info(f"人类文本: {len(human_texts)}")
    logging.info(f"AI生成文本: {len(generated_texts)}")
    
    # 划分训练集、验证集和测试集
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    logging.info(f"训练集: {len(train_df)}")
    logging.info(f"验证集: {len(val_df)}")
    logging.info(f"测试集: {len(test_df)}")
    
    # 保存处理后的数据
    output_dir = os.path.join('Project', 'data', 'processed', 'zh', domain)
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_json(os.path.join(output_dir, 'train.json'), orient='records', force_ascii=False)
    val_df.to_json(os.path.join(output_dir, 'val.json'), orient='records', force_ascii=False)
    test_df.to_json(os.path.join(output_dir, 'test.json'), orient='records', force_ascii=False)
    
    return train_df, val_df, test_df

def create_chinese_data_loaders(train_df, val_df, test_df, tokenizer, batch_size=16, max_length=512):
    """创建数据加载器"""
    train_dataset = ChineseTextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length
    )
    val_dataset = ChineseTextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length
    )
    test_dataset = ChineseTextDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    return train_dataset, val_dataset, test_dataset 