import json
import argparse
import numpy as np
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from local_infer import FastDetectGPT
import torch
import os

def compute_criteria(detector, text):
    crit, ntokens = detector.compute_crit(text)
    # 如果crit是向量，取均值
    if isinstance(crit, torch.Tensor) and crit.numel() > 1:
        crit = crit.mean().item()
    elif isinstance(crit, torch.Tensor):
        crit = crit.item()
    return crit

def main(args):
    detector = FastDetectGPT(args)

    # 遍历文件夹下所有json文件
    for file_name in os.listdir(args.data_path):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(args.data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        human_crits, ai_crits = [], []

        # original（人类）
        for sentence in data.get('original', []):
            try:
                crit = compute_criteria(detector, sentence[0:500])
                human_crits.append(crit)
            except Exception as e:
                print(f"Error processing original in {file_name}: {e}")

        # sampled（AI）
        for sentence in data.get('sampled', []):
            try:
                crit = compute_criteria(detector, sentence[0:500])
                ai_crits.append(crit)
            except Exception as e:
                print(f"Error processing sampled in {file_name}: {e}")

        human_crits = np.array(human_crits)
        ai_crits = np.array(ai_crits)

        print(f"==== {file_name} ====")
        print("真人文本: 样本数", len(human_crits))
        print("AI文本: 样本数", len(ai_crits))
        print("真人文本 criterion 前10个样本：", human_crits[:10])
        print("AI文本 criterion 前10个样本：", ai_crits[:10])
        if len(human_crits) > 0:
            print("mu0 (human):", np.mean(human_crits))
            print("sigma0 (human):", np.std(human_crits))
        else:
            print("mu0 (human): N/A")
            print("sigma0 (human): N/A")
        if len(ai_crits) > 0:
            print("mu1 (AI):", np.mean(ai_crits))
            print("sigma1 (AI):", np.std(ai_crits))
        else:
            print("mu1 (AI): N/A")
            print("sigma1 (AI): N/A")
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="/home/wangdx_lab/cse12213023/fast-detect-gpt/gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="/home/wangdx_lab/cse12213023/fast-detect-gpt/gpt2-xl")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--data_path', type=str, required=True, help="包含json文件的数据目录")
    args = parser.parse_args()

    main(args)