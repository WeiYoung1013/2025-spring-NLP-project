import os
import json
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer, BertTokenizerFast
import numpy as np
from data_processor import GhostbusterDataProcessor

def init_tokenizers():
    """
    初始化中英文BERT分词器
    """
    en_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    zh_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    return en_tokenizer, zh_tokenizer

def preprocess_text(text, language='en'):
    """
    文本预处理
    Args:
        text: 输入文本
        language: 'en' 或 'zh'
    Returns:
        处理后的文本
    """
    if not text:
        return ""
        
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if language == 'en':
        # 英文特定处理
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 保留基本标点符号，只移除特殊符号
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
        # 移除连续的标点符号
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        # 确保标点符号后有空格
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        # 移除纯数字（保留包含字母的数字组合）
        text = re.sub(r'\b\d+\b', '', text)
        # 转换为小写
        text = text.lower()
    else:
        # 中文特定处理
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 移除英文和标点
        text = re.sub(r'[a-zA-Z]+', '', text)
        # 移除多余的标点符号
        text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    
    # 最终的空白处理
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 检查处理后的文本长度
    if len(text.split()) < 3:  # 如果文本太短（少于3个词），返回空字符串
        return ""
    
    return text

def truncate_text(text, tokenizer, max_length=512):
    """
    截断文本以适应BERT的最大长度限制
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  # 考虑[CLS]和[SEP]标记
        tokens = tokens[:(max_length - 2)]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

def check_text_length(text, tokenizer, max_length=512):
    """
    检查文本长度是否适合BERT
    """
    tokens = tokenizer.tokenize(text)
    token_length = len(tokens)
    # 考虑[CLS]和[SEP]标记，所以是max_length-2
    return token_length <= (max_length - 2), token_length

def split_long_text(text, tokenizer, max_length=512):
    """
    将长文本分割成多个短文本片段
    Args:
        text: 输入文本
        tokenizer: 分词器
        max_length: 最大长度（考虑[CLS]和[SEP]标记，实际最大长度为max_length-2）
    Returns:
        文本片段列表
    """
    if not text:
        return []
    
    # 预处理文本
    text = preprocess_text(text, 'en')
    if not text:
        return []
    
    # 获取token列表
    tokens = tokenizer.tokenize(text)
    max_segment_length = max_length - 2  # 留出[CLS]和[SEP]的位置
    
    # 如果文本长度在最大长度以内，直接返回
    if len(tokens) <= max_segment_length:
        return [text]
    
    # 如果文本超长，切分成多个片段
    text_segments = []
    start = 0
    while start < len(tokens):
        # 取出一段token
        segment_tokens = tokens[start:start + max_segment_length]
        # 转换回文本
        segment_text = tokenizer.convert_tokens_to_string(segment_tokens)
        if segment_text:
            text_segments.append(segment_text)
        start += max_segment_length
    
    return text_segments

def load_english_data(data_dir, tokenizer):
    """
    加载英文数据集
    """
    texts = []
    labels = []
    sources = []
    skipped = 0
    empty_count = 0
    segments_count = 0
    original_count = 0
    
    # 遍历所有领域
    domains = ['wp', 'reuter', 'essay']
    for domain in domains:
        domain_dir = os.path.join(data_dir, 'ghostbuster-data', 'ghostbuster-data', domain)
        print(f"\n处理领域: {domain} 从 {domain_dir}")
        domain_texts = 0
        domain_segments = 0
        
        # 加载人类文本
        human_dir = os.path.join(domain_dir, 'human')
        if os.path.exists(human_dir):
            print(f"找到人类文本目录: {human_dir}")
            
            if domain == 'reuter':
                # reuter领域：遍历作者目录
                for author in os.listdir(human_dir):
                    author_dir = os.path.join(human_dir, author)
                    if os.path.isdir(author_dir) and not author.startswith('.') and author != 'logprobs':
                        print(f"处理作者: {author}")
                        for file in tqdm(os.listdir(author_dir), desc=f"加载{domain} {author}的人类文本"):
                            if file.endswith('.txt') and not file.startswith('.'):
                    try:
                                    with open(os.path.join(author_dir, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                            # 分割长文本
                                            text_segments = split_long_text(text, tokenizer)
                                            if text_segments:
                                                original_count += 1
                                                texts.extend(text_segments)
                                                labels.extend([0] * len(text_segments))  # 0表示人类文本
                                                sources.extend([f'{domain}_{author}_human'] * len(text_segments))
                                    domain_texts += 1
                                                if len(text_segments) > 1:
                                                    domain_segments += len(text_segments) - 1
                                                    segments_count += len(text_segments) - 1
                                else:
                                    empty_count += 1
                                    skipped += 1
                            else:
                                empty_count += 1
                                skipped += 1
                    except Exception as e:
                                    print(f"警告: 处理文件出错 {file}: {str(e)}")
                        skipped += 1
        else:
                # essay和wp领域：直接遍历txt文件
                for file in tqdm(os.listdir(human_dir), desc=f"加载{domain}人类文本"):
                    if file.endswith('.txt') and not file.startswith('.'):
                        try:
                            with open(os.path.join(human_dir, file), 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                if text:
                                    # 分割长文本
                                    text_segments = split_long_text(text, tokenizer)
                                    if text_segments:
                                        original_count += 1
                                        texts.extend(text_segments)
                                        labels.extend([0] * len(text_segments))  # 0表示人类文本
                                        sources.extend([f'{domain}_human'] * len(text_segments))
                                        domain_texts += 1
                                        if len(text_segments) > 1:
                                            domain_segments += len(text_segments) - 1
                                            segments_count += len(text_segments) - 1
                                    else:
                                        empty_count += 1
                                        skipped += 1
                                else:
                                    empty_count += 1
                                    skipped += 1
                        except Exception as e:
                            print(f"警告: 处理文件出错 {file}: {str(e)}")
                            skipped += 1
        else:
            print(f"警告: 未找到人类文本目录 {human_dir}")
        
        # 加载AI生成文本
        ai_dirs = ['gpt', 'claude', 'gpt_writing', 'gpt_semantic', 'gpt_prompt1', 'gpt_prompt2']
        for ai_dir_name in ai_dirs:
            ai_dir = os.path.join(domain_dir, ai_dir_name)
            if os.path.exists(ai_dir):
                print(f"找到AI文本目录: {ai_dir}")
                
                if domain == 'reuter':
                    # reuter领域：遍历作者目录
                    for author in os.listdir(ai_dir):
                        author_dir = os.path.join(ai_dir, author)
                        if os.path.isdir(author_dir) and not author.startswith('.') and author != 'logprobs':
                            print(f"处理作者: {author}")
                            for file in tqdm(os.listdir(author_dir), desc=f"加载{domain} {author}的{ai_dir_name}文本"):
                                if file.endswith('.txt') and not file.startswith('.'):
                                    try:
                                        with open(os.path.join(author_dir, file), 'r', encoding='utf-8') as f:
                                            text = f.read().strip()
                                            if text:
                                                # 分割长文本
                                                text_segments = split_long_text(text, tokenizer)
                                                if text_segments:
                                                    original_count += 1
                                                    texts.extend(text_segments)
                                                    labels.extend([1] * len(text_segments))  # 1表示AI生成文本
                                                    sources.extend([f'{domain}_{author}_{ai_dir_name}'] * len(text_segments))
                                                    domain_texts += 1
                                                    if len(text_segments) > 1:
                                                        domain_segments += len(text_segments) - 1
                                                        segments_count += len(text_segments) - 1
                                                else:
                                                    empty_count += 1
                                                    skipped += 1
                                            else:
                                                empty_count += 1
                                                skipped += 1
                                    except Exception as e:
                                        print(f"警告: 处理文件出错 {file}: {str(e)}")
                                        skipped += 1
                else:
                    # essay和wp领域：直接遍历txt文件
                    for file in tqdm(os.listdir(ai_dir), desc=f"加载{domain} {ai_dir_name}文本"):
                        if file.endswith('.txt') and not file.startswith('.'):
                            try:
                                with open(os.path.join(ai_dir, file), 'r', encoding='utf-8') as f:
                                    text = f.read().strip()
                                    if text:
                                        # 分割长文本
                                        text_segments = split_long_text(text, tokenizer)
                                        if text_segments:
                                            original_count += 1
                                            texts.extend(text_segments)
                                            labels.extend([1] * len(text_segments))  # 1表示AI生成文本
                                            sources.extend([f'{domain}_{ai_dir_name}'] * len(text_segments))
                                            domain_texts += 1
                                            if len(text_segments) > 1:
                                                domain_segments += len(text_segments) - 1
                                                segments_count += len(text_segments) - 1
                                        else:
                                            empty_count += 1
                                            skipped += 1
                                    else:
                                        empty_count += 1
                                        skipped += 1
                            except Exception as e:
                                print(f"警告: 处理文件出错 {file}: {str(e)}")
                            skipped += 1
            else:
                print(f"警告: 未找到AI文本目录 {ai_dir}")
        
        print(f"\n{domain}领域统计:")
        print(f"原始文本数: {domain_texts}")
        print(f"切分后新增片段数: {domain_segments}")
    
    print(f"\n英文数据总体统计:")
    print(f"原始文本数: {original_count}")
    print(f"切分后的总片段数: {len(texts)}")
    print(f"新增的片段数: {segments_count}")
    print(f"人类文本片段: {labels.count(0)}")
    print(f"AI生成文本片段: {labels.count(1)}")
    print(f"空文本数: {empty_count}")
    print(f"跳过的文件: {skipped}")
    
    # 计算每个来源的数量
    source_counts = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\n来源分布:")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count}")
    
    return texts, labels, sources

def process_text(text, source_name, tokenizer):
    """处理单个文本，如果文本过长则切分成多个片段
    Args:
        text: 输入文本
        source_name: 数据来源名称
        tokenizer: BERT分词器
    Returns:
        处理后的文本列表（可能包含多个片段）
    """
    if not text or not isinstance(text, str):
        return []
    
    # 预处理文本
    text = preprocess_text(text, 'zh')
    if not text:
        return []
    
    # 获取token列表
    tokens = tokenizer.tokenize(text)
    max_length = 510  # 留出[CLS]和[SEP]的位置
    
    # 如果文本长度在最大长度以内，直接返回
    if len(tokens) <= max_length:
        return [text]
    
    # 如果文本超长，切分成多个片段
    text_segments = []
    start = 0
    while start < len(tokens):
        # 取出一段token
        segment_tokens = tokens[start:start + max_length]
        # 转换回文本
        segment_text = tokenizer.convert_tokens_to_string(segment_tokens)
        if segment_text:
            # 对中文文本进行额外的清理
            segment_text = re.sub(r'\s+', '', segment_text)  # 移除空白字符
            segment_text = re.sub(r'##', '', segment_text)   # 移除BERT分词产生的##标记
        if segment_text:
            text_segments.append(segment_text)
        start += max_length
    
    return text_segments

def load_chinese_data(data_dir, tokenizer):
    """
    加载中文数据集（face2_zh_json）
    Args:
        data_dir: 数据根目录
        tokenizer: BERT中文分词器
    Returns:
        texts: 文本列表
        labels: 标签列表（0为人类，1为AI）
        sources: 来源列表
    """
    texts = []
    labels = []
    sources = []
    skipped = 0
    empty_count = 0
    segments_count = 0
    original_count = 0
    
    # 加载人类文本
    human_dir = os.path.join(data_dir, 'face2_zh_json', 'human', 'zh_unicode')
    if os.path.exists(human_dir):
        print(f"处理人类文本目录: {human_dir}")
        human_files = ['news-zh.json', 'webnovel.json', 'wiki-zh.json']
        
        for file in tqdm(human_files, desc="加载人类文本"):
            file_path = os.path.join(human_dir, file)
            source_name = f'human_{os.path.splitext(file)[0]}'
            
            if os.path.exists(file_path):
                try:
                with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    # 优先使用output字段
                                    text = item.get('output', '')
                                    if not text:
                                        # 如果output为空，使用input字段
                                        text = item.get('input', '')
                                    
                                    text_segments = process_text(text, source_name, tokenizer)
                                    if text_segments:
                                        original_count += 1
                                        texts.extend(text_segments)
                                        labels.extend([0] * len(text_segments))
                                        sources.extend([source_name] * len(text_segments))
                                        if len(text_segments) > 1:
                                            segments_count += len(text_segments) - 1
                                    else:
                                        empty_count += 1
                    except Exception as e:
                    print(f"警告: 处理文件出错 {file}: {str(e)}")
                        skipped += 1
    
    # 加载AI生成文本
    generated_dir = os.path.join(data_dir, 'face2_zh_json', 'generated', 'zh_qwen2')
    if os.path.exists(generated_dir):
        print(f"\n处理AI生成文本目录: {generated_dir}")
        for file in tqdm(os.listdir(generated_dir), desc="加载AI生成文本"):
            if file.endswith('.json'):
                file_path = os.path.join(generated_dir, file)
                try:
                with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'output' in data:
                            for key, text in data['output'].items():
                                text_segments = process_text(text, 'generated_zh_qwen2', tokenizer)
                                if text_segments:
                                    original_count += 1
                                    texts.extend(text_segments)
                                    labels.extend([1] * len(text_segments))
                                    sources.extend(['generated_zh_qwen2'] * len(text_segments))
                                    if len(text_segments) > 1:
                                        segments_count += len(text_segments) - 1
                                else:
                                    empty_count += 1
                    except Exception as e:
                    print(f"警告: 处理文件出错 {file}: {str(e)}")
                        skipped += 1
    
    print(f"\n数据处理统计:")
    print(f"原始文档数: {original_count}")
    print(f"切分后的总片段数: {len(texts)}")
    print(f"新增的片段数: {segments_count}")
    print(f"人类文本片段: {labels.count(0)}")
    print(f"AI生成文本片段: {labels.count(1)}")
    print(f"空文本数: {empty_count}")
    print(f"跳过的文件: {skipped}")
    
    # 显示每个来源的数量
    source_counts = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    print("\n来源分布:")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count}")
    
    return texts, labels, sources

def save_data(texts, labels, sources, output_dir, language):
    """
    保存处理后的数据，按照领域分别存储
    """
    # 创建语言特定的输出目录
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'source': sources
    })
    
    # 显示总体数据分布
    print(f"\n{language.upper()} 数据总体分布:")
    print(f"总样本数: {len(df)}")
    print(f"人类文本数: {len(df[df['label'] == 0])}")
    print(f"AI生成文本数: {len(df[df['label'] == 1])}")
    print("\n来源分布:")
    print(df['source'].value_counts())
    
    # 按领域分割数据
    domains = ['essay', 'reuter', 'wp']
    for domain in domains:
        # 筛选该领域的数据
        domain_df = df[df['source'].str.contains(domain)]
        if len(domain_df) == 0:
            print(f"\n警告: {domain}领域没有数据")
            continue
            
        # 为该领域创建目录
        domain_dir = os.path.join(lang_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
    
    # 划分训练集和测试集
    train_df, test_df = train_test_split(
            domain_df, 
        test_size=0.2, 
        random_state=42,
            stratify=domain_df['label']
    )
    
    # 保存数据
        train_df.to_json(os.path.join(domain_dir, 'train.json'), orient='records', force_ascii=False, indent=2)
        test_df.to_json(os.path.join(domain_dir, 'test.json'), orient='records', force_ascii=False, indent=2)
    
        # 保存该领域的数据统计信息
        with open(os.path.join(domain_dir, 'stats.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{domain.upper()} 领域数据分布:\n")
            f.write(f"总样本数: {len(domain_df)}\n")
            f.write(f"人类文本数: {len(domain_df[domain_df['label'] == 0])}\n")
            f.write(f"AI生成文本数: {len(domain_df[domain_df['label'] == 1])}\n")
            f.write("\n来源分布:\n")
            f.write(domain_df['source'].value_counts().to_string())
            f.write(f"\n\n训练集大小: {len(train_df)}\n")
            f.write(f"测试集大小: {len(test_df)}\n")
            
        print(f"\n{domain}领域数据已保存到: {domain_dir}")
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    # 初始化分词器
    en_tokenizer, zh_tokenizer = init_tokenizers()
    
    # 处理英文数据
    print("\n处理英文文本数据...")
    en_texts, en_labels, en_sources = load_english_data(data_dir, en_tokenizer)
    
    # 保存英文处理后的数据
    print("\n保存英文数据...")
    save_data(en_texts, en_labels, en_sources, output_dir, 'en')
    
    # 处理中文数据
    print("\n处理中文文本数据...")
    zh_texts, zh_labels, zh_sources = load_chinese_data(data_dir, zh_tokenizer)
    
    # 保存中文处理后的数据
    print("\n保存中文数据...")
    save_data(zh_texts, zh_labels, zh_sources, output_dir, 'zh')
    
    print("\n数据预处理完成!")
    print(f"英文数据保存到: {os.path.join(output_dir, 'en')}")
    print(f"中文数据保存到: {os.path.join(output_dir, 'zh')}")

if __name__ == '__main__':
    main() 