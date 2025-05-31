import json
import os

def check_json_file(file_path):
    """检查JSON文件的格式"""
    print(f"检查文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前5行
            print("\n前5行内容:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"第{i+1}行: {line.strip()}")
                else:
                    break
            
            # 重置文件指针
            f.seek(0)
            
            # 尝试解析整个文件
            data = json.load(f)
            print("\n文件格式:")
            if isinstance(data, list):
                print(f"- 是列表，长度: {len(data)}")
                if data:
                    print("- 第一个元素的键:")
                    for key in data[0].keys():
                        print(f"  - {key}")
            elif isinstance(data, dict):
                print("- 是字典")
                print("- 顶层键:")
                for key in data.keys():
                    print(f"  - {key}")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_file = os.path.join(base_dir, 'data', 'face2_zh_json', 'human', 'zh_unicode', 'news-zh.json')
    check_json_file(json_file) 