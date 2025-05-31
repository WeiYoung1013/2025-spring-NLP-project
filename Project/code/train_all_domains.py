import os
import torch
import argparse
from train import train_model, evaluate_model, set_seed
from models import TextDetectionModel
from transformers import BertTokenizer, BertConfig
from data_processor import load_data, TextDataset, create_data_loaders
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import logging
import sys
from pathlib import Path

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Project/data/processed/en',
                       help='英文数据根目录')
    parser.add_argument('--output_dir', type=str, default='Project/output',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='预训练模型名称')
    return parser.parse_args()

def train_domain_model(args, domain, device):
    """训练单个领域的模型
    
    Args:
        args: 参数
        domain: 领域名称
        device: 设备
    """
    logging.info(f"\n开始训练 {domain} 领域模型...")
    
    try:
        # 创建领域特定的输出目录
        domain_output_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # 加载数据
        logging.info(f"加载 {domain} 领域数据...")
        train_df, val_df, test_df = load_data(args.data_dir, domain)
        
        # 创建分词器和模型
        logging.info("初始化分词器和模型...")
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        config = BertConfig.from_pretrained(args.model_name)
        model = TextDetectionModel(config)
        model = model.to(device)
        
        # 创建数据集
        train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, args.max_length)
        val_dataset = TextDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, args.max_length)
        test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, args.max_length)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, args.batch_size
        )
        
        # 计算总训练步数
        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        # 创建优化器
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练模型
        logging.info("开始训练过程...")
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            args.num_epochs,
            domain_output_dir
        )
        
        # 加载最佳模型
        best_model_path = os.path.join(domain_output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"加载最佳模型: {best_model_path}")
        else:
            logging.warning(f"未找到最佳模型文件: {best_model_path}")
        
        # 在测试集上评估
        logging.info("在测试集上评估模型...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        logging.info(f"\n{domain} 领域测试集结果:")
        for metric_name, value in test_metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
        
        # 保存测试结果
        test_results = {
            'metrics': test_metrics,
            'best_val_f1': checkpoint.get('best_val_f1', None),
            'best_epoch': checkpoint.get('epoch', None),
            'val_metrics': checkpoint.get('val_metrics', None)
        }
        
        results_file = os.path.join(domain_output_dir, "test_results.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=4)
        logging.info(f"测试结果已保存到: {results_file}")
        
        return model, tokenizer, test_metrics
        
    except Exception as e:
        logging.error(f"训练 {domain} 领域模型时发生错误: {str(e)}")
        raise

def perform_immediate_ood_detection(source_model, source_tokenizer, source_domain, args, device, domains):
    """在单个模型训练完成后立即进行OOD检测
    
    Args:
        source_model: 源域模型
        source_tokenizer: 源域分词器
        source_domain: 源域名称
        args: 参数
        device: 设备
        domains: 所有域的列表
    """
    logging.info(f"\n对 {source_domain} 模型执行OOD检测...")
    ood_results = {}
    
    try:
        source_model.eval()
        
        # 在其他域上进行评估
        for target_domain in domains:
            if target_domain != source_domain:
                logging.info(f"在 {target_domain} 域上测试")
                
                # 加载目标域数据
                _, _, test_df = load_data(args.data_dir, target_domain)
                test_dataset = TextDataset(
                    test_df['text'].tolist(),
                    test_df['label'].tolist(),
                    source_tokenizer,
                    args.max_length
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4
                )
                
                # 评估
                metrics = evaluate_model(source_model, test_loader, device)
                ood_results[target_domain] = metrics
                
                logging.info(f"在 {target_domain} 域上的性能:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")
        
        # 保存当前模型的OOD结果
        ood_results_file = os.path.join(args.output_dir, source_domain, "ood_results.json")
        with open(ood_results_file, "w") as f:
            json.dump(ood_results, f, indent=4)
        logging.info(f"OOD检测结果已保存到: {ood_results_file}")
        
        return ood_results
        
    except Exception as e:
        logging.error(f"执行OOD检测时发生错误: {str(e)}")
        raise

def main():
    args = parse_args()
    
    try:
        # 设置随机种子
        set_seed(42)
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 训练所有领域的模型
        domains = ['essay', 'reuter', 'wp']
        all_ood_results = {}
        domain_metrics = {}
        
        for domain in domains:
            # 训练当前域的模型
            model, tokenizer, metrics = train_domain_model(args, domain, device)
            domain_metrics[domain] = metrics
            
            # 立即进行OOD检测
            ood_results = perform_immediate_ood_detection(
                model, tokenizer, domain, args, device, domains
            )
            all_ood_results[domain] = ood_results
            
            # 释放GPU内存
            del model
            torch.cuda.empty_cache()
        
        # 保存所有结果
        final_results = {
            'domain_metrics': domain_metrics,
            'ood_results': all_ood_results,
            'training_args': vars(args)
        }
        
        results_file = os.path.join(args.output_dir, "all_results.json")
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=4)
        logging.info(f"所有结果已保存到: {results_file}")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 