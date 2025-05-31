import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer, BertTokenizerFast
import logging
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TextPreprocessor:
    def __init__(self, language: str = 'en'):
        """
        初始化文本预处理器
        Args:
            language: 'en' 或 'zh'
        """
        self.language = language
        self.tokenizer = self._init_tokenizer()
        self.max_length = 512  # BERT的最大长度限制

    def _init_tokenizer(self):
        """初始化对应语言的分词器"""
        if self.language == 'en':
            return BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            return BertTokenizerFast.from_pretrained('bert-base-chinese')

    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        Args:
            text: 输入文本
        Returns:
            处理后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 基础清理
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        if self.language == 'en':
            # 英文特定处理
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
            text = re.sub(r'([.,!?])\1+', r'\1', text)
        else:
            # 中文特定处理
            text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]', '', text)
            text = re.sub(r'\s+', '', text)

        return text.strip()

    def truncate_and_pad(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        截断和填充文本
        Args:
            text: 输入文本
        Returns:
            包含input_ids和attention_mask的字典，如果文本无效则返回None
        """
        if not text:
            return None

        # 使用tokenizer进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 预处理文本
        processed_text = self.preprocessor.preprocess_text(text)
        encoded = self.preprocessor.truncate_and_pad(processed_text)

        if encoded is None:
            # 如果文本处理失败，返回一个空序列
            return {
                'input_ids': torch.zeros(self.preprocessor.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.preprocessor.max_length, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    def __init__(self, base_dir: str):
        """
        初始化数据处理器
        Args:
            base_dir: 数据根目录
        """
        self.base_dir = base_dir
        self.en_preprocessor = TextPreprocessor('en')
        self.zh_preprocessor = TextPreprocessor('zh')

    def process_english_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        处理英文数据集
        Returns:
            texts: 文本列表
            labels: 标签列表
            sources: 来源列表
        """
        texts, labels, sources = [], [], []
        ghostbuster_dir = os.path.join(self.base_dir, 'ghostbuster-data', 'ghostbuster-data')
        
        # 处理不同领域
        domains = ['wp', 'reuter', 'essay']
        for domain in domains:
            domain_dir = os.path.join(ghostbuster_dir, domain)
            if not os.path.exists(domain_dir):
                logging.warning(f"Domain directory not found: {domain_dir}")
                continue
                
            # 处理人类文本
            human_dir = os.path.join(domain_dir, 'human')
            if os.path.exists(human_dir):
                for root, _, files in os.walk(human_dir):
                    for file in tqdm(files, desc=f"Processing {domain} human texts"):
                        if file.endswith('.txt') and not file.endswith('-ada.txt') and not file.endswith('-davinci.txt'):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                    text = f.read().strip()
                                    if text:
                                        processed_text = self.en_preprocessor.preprocess_text(text)
                                        if processed_text:
                                            texts.append(processed_text)
                                            labels.append(0)
                                            sources.append(f"{domain}_human")
                            except Exception as e:
                                logging.warning(f"Error processing {file}: {str(e)}")
            else:
                logging.warning(f"Human directory not found: {human_dir}")

            # 处理所有AI生成文本类型
            ai_types = ['claude', 'gpt', 'gpt_prompt1', 'gpt_prompt2', 'gpt_semantic', 'gpt_writing']
            for ai_type in ai_types:
                ai_dir = os.path.join(domain_dir, ai_type)
                if os.path.exists(ai_dir):
                    for root, _, files in os.walk(ai_dir):
                        if 'logprobs' in root:  # 跳过logprobs目录
                            continue
                        for file in tqdm(files, desc=f"Processing {domain} {ai_type} texts"):
                            if file.endswith('.txt') and not file.endswith('-ada.txt') and not file.endswith('-davinci.txt'):
                                try:
                                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                        text = f.read().strip()
                                        if text:
                                            processed_text = self.en_preprocessor.preprocess_text(text)
                                            if processed_text:
                                                texts.append(processed_text)
                                                labels.append(1)
                                                sources.append(f"{domain}_{ai_type}")
                                except Exception as e:
                                    logging.warning(f"Error processing {file}: {str(e)}")
                else:
                    logging.warning(f"AI directory not found: {ai_dir}")

        if not texts:
            logging.warning("No English texts were processed!")
            
        return texts, labels, sources

    def process_chinese_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        处理中文数据集
        Returns:
            texts: 文本列表
            labels: 标签列表
            sources: 来源列表
        """
        texts, labels, sources = [], [], []
        face2_zh_dir = os.path.join(self.base_dir, 'face2_zh_json')
        
        if not os.path.exists(face2_zh_dir):
            logging.warning(f"Chinese data directory not found: {face2_zh_dir}")
            return texts, labels, sources

        # 处理人类文本
        human_dir = os.path.join(face2_zh_dir, 'human')
        if os.path.exists(human_dir):
            for root, _, files in os.walk(human_dir):
                for file in tqdm(files, desc="Processing human Chinese texts"):
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            text = item.get('output', '') or item.get('input', '')
                                            if text:
                                                processed_text = self.zh_preprocessor.preprocess_text(text)
                                                if processed_text:
                                                    texts.append(processed_text)
                                                    labels.append(0)
                                                    sources.append(f"human_{os.path.splitext(file)[0]}")
                        except Exception as e:
                            logging.warning(f"Error processing {file}: {str(e)}")
        else:
            logging.warning(f"Human directory not found: {human_dir}")

        # 处理AI生成文本
        generated_dir = os.path.join(face2_zh_dir, 'generated')
        if os.path.exists(generated_dir):
            for root, _, files in os.walk(generated_dir):
                for file in tqdm(files, desc="Processing AI Chinese texts"):
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict) and 'output' in data:
                                    for text in data['output'].values():
                                        if text:
                                            processed_text = self.zh_preprocessor.preprocess_text(text)
                                            if processed_text:
                                                texts.append(processed_text)
                                                labels.append(1)
                                                sources.append('generated_text')
                        except Exception as e:
                            logging.warning(f"Error processing {file}: {str(e)}")
        else:
            logging.warning(f"Generated directory not found: {generated_dir}")

        if not texts:
            logging.warning("No Chinese texts were processed!")
            
        return texts, labels, sources

    def save_processed_data(self, texts: List[str], labels: List[int], sources: List[str], 
                           output_dir: str, language: str):
        """
        保存处理后的数据
        """
        if not texts:
            logging.warning(f"No {language} texts to save!")
            return
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'source': sources
        })

        # 显示数据统计
        logging.info(f"\n{language.upper()} 数据集统计:")
        logging.info(f"总样本数: {len(df)}")
        logging.info(f"人类文本数: {len(df[df['label'] == 0])}")
        logging.info(f"AI生成文本数: {len(df[df['label'] == 1])}")
        logging.info("\n来源分布:")
        logging.info(df['source'].value_counts())

        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )

        # 保存数据
        train_df.to_json(os.path.join(output_dir, 'train.json'), 
                        orient='records', force_ascii=False, indent=2)
        test_df.to_json(os.path.join(output_dir, 'test.json'), 
                        orient='records', force_ascii=False, indent=2)

        # 保存数据统计信息
        with open(os.path.join(output_dir, 'stats.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{language.upper()} 数据集统计\n")
            f.write(f"总样本数: {len(df)}\n")
            f.write(f"人类文本数: {len(df[df['label'] == 0])}\n")
            f.write(f"AI生成文本数: {len(df[df['label'] == 1])}\n")
            f.write("\n来源分布:\n")
            f.write(df['source'].value_counts().to_string())
            f.write(f"\n\n训练集大小: {len(train_df)}\n")
            f.write(f"测试集大小: {len(test_df)}\n")
            
        logging.info(f"\n数据已保存至: {output_dir}")
        logging.info(f"训练集大小: {len(train_df)}")
        logging.info(f"测试集大小: {len(test_df)}")

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'data', 'processed')

    # 创建数据处理器
    processor = DataProcessor(data_dir)

    # 处理英文数据
    logging.info("处理英文数据...")
    en_texts, en_labels, en_sources = processor.process_english_data()
    en_output_dir = os.path.join(output_dir, 'en')
    processor.save_processed_data(en_texts, en_labels, en_sources, en_output_dir, 'en')

    # 处理中文数据
    logging.info("\n处理中文数据...")
    zh_texts, zh_labels, zh_sources = processor.process_chinese_data()
    zh_output_dir = os.path.join(output_dir, 'zh')
    processor.save_processed_data(zh_texts, zh_labels, zh_sources, zh_output_dir, 'zh')

    logging.info("\n数据处理完成!")
    logging.info(f"英文数据保存至: {en_output_dir}")
    logging.info(f"中文数据保存至: {zh_output_dir}")

if __name__ == '__main__':
    main() 