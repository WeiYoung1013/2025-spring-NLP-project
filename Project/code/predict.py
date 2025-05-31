import torch
import argparse
from model import get_model_and_tokenizer

def predict_text(text, model, tokenizer, device, max_length=512):
    """
    对单个文本进行预测
    
    Args:
        text: 输入文本
        model: 加载的模型
        tokenizer: 分词器
        device: 设备
        max_length: 最大序列长度
    
    Returns:
        预测标签和概率
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 对文本进行编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 将数据移到指定设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
    
    return predicted_label, confidence

def load_model(model_path, language, device):
    """
    加载保存的模型
    """
    # 获取模型和分词器
    model, tokenizer = get_model_and_tokenizer(language)
    
    # 加载模型权重
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    print("\nModel state dict keys:")
    for key in checkpoint['model_state_dict'].keys():
        print(key)
    
    print("\nCurrent model state dict keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移到指定设备
    model = model.to(device)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测文本是人工生成还是人类撰写')
    parser.add_argument('--model_path', type=str, required=True, help='保存的模型路径')
    parser.add_argument('--language', type=str, choices=['en', 'zh'], required=True, help='语言')
    parser.add_argument('--text', type=str, required=True, help='要预测的文本')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.language, device)
    
    # 进行预测
    label, confidence = predict_text(args.text, model, tokenizer, device, args.max_length)
    
    # 输出结果
    print("\n预测结果:")
    print(f"输入文本: {args.text}")
    print(f"预测标签: {'人工生成' if label == 1 else '人类撰写'}")
    print(f"置信度: {confidence:.4f}")

if __name__ == '__main__':
    main()