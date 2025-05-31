import os
import torch
import argparse
import logging
import json
from models import TextDetectionModel
from transformers import BertTokenizer, BertConfig
from data_processor import load_data, TextDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Project/data/processed/en',
                       help='英文数据根目录')
    parser.add_argument('--model_dir', type=str, default='Project/output/training_results',
                       help='模型目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    return parser.parse_args()

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs['logits']
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(all_labels, all_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }

def evaluate_domain(args, domain, device):
    """评估单个领域的模型"""
    logging.info(f"\n评估 {domain} 领域模型...")
    
    # 加载模型
    model_path = os.path.join(args.model_dir, domain, 'best_model.pt')
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = TextDetectionModel(config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载数据
    train_df, val_df, test_df = load_data(args.data_dir, domain)
    
    # 创建数据集和加载器
    test_dataset = TextDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        args.max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 在测试集上评估
    metrics = evaluate_model(model, test_loader, device)
    logging.info(f"\n{domain} 领域测试集结果:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
    
    # 执行OOD评估
    ood_results = {}
    domains = ['essay', 'reuter', 'wp']
    
    for target_domain in domains:
        if target_domain != domain:
            logging.info(f"\n在 {target_domain} 域上测试...")
            
            # 加载目标域数据
            _, _, target_test_df = load_data(args.data_dir, target_domain)
            target_dataset = TextDataset(
                target_test_df['text'].tolist(),
                target_test_df['label'].tolist(),
                tokenizer,
                args.max_length
            )
            target_loader = DataLoader(
                target_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # 评估
            target_metrics = evaluate_model(model, target_loader, device)
            ood_results[target_domain] = target_metrics
            
            logging.info(f"在 {target_domain} 域上的性能:")
            for metric_name, value in target_metrics.items():
                logging.info(f"{metric_name}: {value:.4f}")
    
    return metrics, ood_results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    domains = ['essay', 'reuter', 'wp']
    all_results = {}
    
    for domain in domains:
        domain_metrics, ood_results = evaluate_domain(args, domain, device)
        all_results[domain] = {
            'in_domain_metrics': domain_metrics,
            'ood_metrics': ood_results
        }
    
    # 保存结果
    output_file = os.path.join(args.model_dir, 'detailed_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"\n详细评估结果已保存到: {output_file}")

if __name__ == '__main__':
    main() 