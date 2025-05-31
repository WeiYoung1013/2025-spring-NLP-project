import json
import os

def check_json_format(file_path):
    """检查JSON文件的格式"""
    print(f"\n{'='*50}")
    print(f"检查文件: {file_path}")
    print(f"{'='*50}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 打印数据类型
        print(f"数据类型: {type(data)}")
        
        # 如果是列表，打印详细信息
        if isinstance(data, list):
            print(f"\n总样本数: {len(data)}")
            print("\n第一个元素的结构:")
            first_item = data[0]
            for key, value in first_item.items():
                print(f"- {key}: {type(value)}")
                if isinstance(value, str):
                    print(f"  长度: {len(value)}")
                    print(f"  预览: {value[:100]}...")
        
        # 如果是字典，打印详细信息
        elif isinstance(data, dict):
            print("\n字典结构:")
            for key, value in data.items():
                print(f"\n键: {key}")
                print(f"值类型: {type(value)}")
                if isinstance(value, list):
                    print(f"列表长度: {len(value)}")
                    if len(value) > 0:
                        print("第一个元素类型:", type(value[0]))
                        if isinstance(value[0], str):
                            print(f"第一个元素预览: {value[0][:100]}...")
                elif isinstance(value, str):
                    print(f"字符串长度: {len(value)}")
                    print(f"预览: {value[:100]}...")
    
    except Exception as e:
        print(f"错误: {str(e)}")

def main():
    # 检查人类文本
    human_dir = "Project/data/face2_zh_json/human/zh_unicode"
    for filename in ["news-zh.json", "webnovel.json", "wiki-zh.json"]:
        check_json_format(os.path.join(human_dir, filename))
    
    # 检查生成文本
    generated_dir = "Project/data/face2_zh_json/generated/zh_qwen2"
    for filename in ["news-zh.qwen2-72b-base.json", "webnovel.qwen2-72b-base.json", "wiki-zh.qwen2-72b-base.json"]:
        check_json_format(os.path.join(generated_dir, filename))

if __name__ == "__main__":
    main() 