# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            '/home/wangdx_lab/cse12213023/fast-detect-gpt/gpt2-xl_/home/wangdx_lab/cse12213023/fast-detect-gpt/gpt2-xl': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            '/home/wangdx_lab/cse12213023/2025-spring-NLP-project/Project/rbtl3_/home/wangdx_lab/cse12213023/2025-spring-NLP-project/Project/rbtl3':{'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            '/home/wangdx_lab/cse12213023/2025-spring-NLP-project/Project/rwel_/home/wangdx_lab/cse12213023/2025-spring-NLP-project/Project/rwel':{'mu0': -12.4, 'sigma0':  3.6, 'mu1': -11.8, 'sigma1': 3.5},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


def run(args, data_dirs, output_path="detectgpt_results_rest.txt"):
    detector = FastDetectGPT(args)
    print('Batch demo for Fast-DetectGPT')
    print(f'Saving results to: {output_path}\n')

    # 区间：[0,0.1), [0.1,0.2), ..., [0.9,1.0]
    bins = [i/10 for i in range(11)]
    bin_labels = [f"[{int(bins[i]*100)}-{int(bins[i+1]*100)})" for i in range(10)]
    bin_labels[-1] = "[90-100]"

    stats_per_dir = {}

    with open(output_path, "w", encoding='utf-8') as txtfile:
        for data_dir in data_dirs:
            txtfile.write(f"\n=== Processing folder: {data_dir} ===\n")
            bin_counts = [0]*10
            total = 0

            for root, dirs, files in os.walk(data_dir):
                if 'logprob' in root:
                    continue
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            prob, crit, ntokens = detector.compute_prob(text)
                            idx = min(int(prob*10), 9)
                            bin_counts[idx] += 1
                            total += 1
                        except Exception as e:
                            txtfile.write(f"Error processing {file_path}: {e}\n")
            # 汇总
            txtfile.write("\n概率区间分布统计:\n")
            for i, count in enumerate(bin_counts):
                percent = (count/total*100) if total > 0 else 0
                txtfile.write(f"{bin_labels[i]}: {count} ({percent:.2f}%)\n")
            txtfile.write(f"总文本数: {total}\n")
            txtfile.write("="*50 + "\n")

    print("Finished. Results saved.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b")
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    data_dirs=["/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/essay/claude",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/gpt",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/gpt-prompt1",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/gpt-prompt2",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/gpt-semantic",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/gpt-writing",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/reuter/human",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/gpt",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/gpt-prompt1",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/gpt-prompt2",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/gpt-semantic",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/gpt-writing",
        "/home/wangdx_lab/cse12213023/fast-detect-gpt/ghostbuster-data/wp/human"
    ]
    run(args, data_dirs)



