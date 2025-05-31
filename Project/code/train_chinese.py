import os
import torch
import argparse
import logging
import sys
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from models import TextDetectionModel
from chinese_data_processor import process_chinese_data, create_chinese_data_loaders
from train import train_model, evaluate_model, set_seed
import json
from pathlib import Path
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chinese_training.log')
    ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_dir', type=str, default='Project/data/face2_zh_json/human/zh_unicode',
                      help='人类写作文本目录')
    parser.add_argument('--generated_dir', type=str, default='Project/data/face2_zh_json/generated/zh_qwen2',
                      help='AI生成文本目录')
    parser.add_argument('--output_dir', type=str, default='Project/output2/chinese_results',
                      help='输出目录')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese',
                      help='预训练模型名称')
    parser.add_argument('--batch_size', type=int, default=48,
                      help='批次大小，由于使用了较小的模型，可以使用更大的batch size')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='由于batch size更大，适当增加学习率')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    return parser.parse_args()

def check_paths(args):
    """检查所有必要的路径是否存在"""
    paths = {
        '人类文本目录': Path(args.human_dir),
        'AI生成文本目录': Path(args.generated_dir)
    }
    
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 不存在: {path}")
        logging.info(f"已找到 {name}: {path}")

def train_chinese_domain(args, domain, device, all_domains):
    """训练单个中文领域的模型并进行OOD测试
    
    Args:
        args: 参数
        domain: 当前训练的领域
        device: 设备
        all_domains: 所有领域列表
    """
    try:
        logging.info(f"\n{'='*50}")
        logging.info(f"开始训练 {domain} 领域模型...")
        logging.info(f"{'='*50}")
        
        # 创建领域特定的输出目录
        domain_output_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # 处理当前领域数据
        train_df, val_df, test_df = process_chinese_data(
            args.human_dir,
            args.generated_dir,
            domain
        )
        
        logging.info("正在初始化模型和tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        config = BertConfig.from_pretrained(args.model_name)
        config.hidden_dropout_prob = 0.2
        config.attention_probs_dropout_prob = 0.2
        model = TextDetectionModel(config)
        model = model.to(device)
        
        # 创建数据加载器
        train_dataset, val_dataset, test_dataset = create_chinese_data_loaders(
            train_df, val_df, test_df,
            tokenizer,
            args.batch_size,
            args.max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 训练模型
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.num_epochs * len(train_loader) * args.warmup_ratio),
            num_training_steps=len(train_loader) * args.num_epochs
        )
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=domain_output_dir,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # 在测试集上评估
        logging.info(f"\n在 {domain} 测试集上评估...")
        test_metrics = evaluate_model(trained_model, test_loader, device)
        
        # 进行OOD测试
        ood_results = {}
        for target_domain in all_domains:
            if target_domain != domain:
                logging.info(f"\n在 {target_domain} 域上进行OOD测试...")
                
                # 加载目标域数据
                _, _, target_test_df = process_chinese_data(
                    args.human_dir,
                    args.generated_dir,
                    target_domain
                )
                
                # 创建目标域测试集
                _, _, target_test_dataset = create_chinese_data_loaders(
                    target_test_df, target_test_df, target_test_df,  # 只需要测试集
                    tokenizer,
                    args.batch_size,
                    args.max_length
                )
                
                target_test_loader = DataLoader(
                    target_test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False
                )
                
                # 评估OOD性能
                ood_metrics = evaluate_model(trained_model, target_test_loader, device)
                ood_results[target_domain] = ood_metrics
        
        # 保存结果
        results = {
            'test_metrics': test_metrics,
            'ood_results': ood_results,
            'training_params': vars(args)
        }
        
        results_file = os.path.join(domain_output_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logging.info(f"结果已保存到: {results_file}")
        
        return trained_model, test_metrics, ood_results
        
    except Exception as e:
        logging.error(f"{domain} 领域训练失败: {str(e)}", exc_info=True)
        raise

def main():
    args = parse_args()
    try:
        # 设置随机种子
        set_seed(42)
        
        # 检查路径
        check_paths(args)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"输出目录: {args.output_dir}")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        
        # 中文数据集的三个领域
        domains = ['news', 'webnovel', 'wiki']
        all_results = {}
        
        # 训练每个领域的模型
        for domain in domains:
            logging.info(f"\n开始处理 {domain} 领域...")
            logging.info(f"人类文本目录: {args.human_dir}")
            logging.info(f"AI生成文本目录: {args.generated_dir}")
            
            trained_model, test_metrics, ood_results = train_chinese_domain(
                args, domain, device, domains
            )
            
            all_results[domain] = {
                'test_metrics': test_metrics,
                'ood_results': ood_results
            }
            
            # 释放GPU内存
            del trained_model
            torch.cuda.empty_cache()
        
        # 保存所有结果
        final_results_file = os.path.join(args.output_dir, 'all_results.json')
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        logging.info(f"\n所有结果已保存到: {final_results_file}")
        
        logging.info("\n所有领域训练完成！")
        
    except Exception as e:
        logging.error(f"训练过程失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 