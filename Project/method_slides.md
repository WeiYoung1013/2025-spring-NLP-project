---
marp: true
theme: default
paginate: true
---

# 中英文AI生成文本检测方法
## 基于BERT的跨语言检测模型

---

# 目录

1. 数据预处理
2. 预训练模型选择
3. 模型架构设计
4. 训练策略
5. 创新亮点

---

# 1. 数据预处理

## 1.1 中文数据
- **数据来源**
  - 人类撰写文本 (human_dir)
  - AI生成文本 (generated_dir)
  - 三大领域：新闻、网络小说、维基百科

## 1.2 英文数据
- **领域覆盖**
  - Reuters新闻
  - Web Prose (WP)
  - Essay论文

---

# 数据处理流程

![数据处理流程 width:800px](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuNBCoKnELT2rKt3AJx9IS2mjoKZDAybCJYp9pCzJ24ejB4qjBk42oYde0000)

- 文本清洗和规范化
- 数据集平衡（1:1）
- 8:1:1划分（训练：验证：测试）
- 标签编码（人类：0，AI：1）

---

# 2. 预训练模型选择

## 2.1 中文模型
- **基础模型**: `hfl/rbtl3`
  - 中文优化
  - 优秀语义理解
  - 适中模型规模

## 2.2 英文模型
- **基础模型**: `bert-base-uncased`
  - 稳定可靠
  - 出色文本理解
  - 资源需求适中

---

# 3. 模型架构设计

![模型架构 width:900px](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuNBCoKnELT2rKt3AJx9IS2mjoKZDAybCJYp9pCzJ24ejB4qjBk42oYde0000)

---

# 模型核心组件

```python
class TextDetectionModel(BertPreTrainedModel):
    def __init__(self, config):
        # BERT基础层
        self.bert = BertModel(config)
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512)
        )
        
        # 分类器
        self.classifier = nn.Linear(256, 2)
```

---

# 创新设计

## 1. 残差连接
- 增强特征提取
- 缓解梯度消失
- 提升训练稳定性

## 2. 多层特征提取
- 双层线性变换
- LayerNorm归一化
- GELU激活函数

## 3. 防过拟合
- Dropout层设计
- 标签平滑技术
- 梯度裁剪策略

---

# 4. 训练策略

## 超参数配置

| 参数 | 中文模型 | 英文模型 |
|------|----------|----------|
| batch_size | 16 | 16 |
| epochs | 10 | 5 |
| learning_rate | 2e-5 | 2e-5 |
| max_length | 512 | 512 |
| warmup_ratio | 0.1 | 0.1 |

---

# 优化策略

## 1. 学习率调度
- 线性预热
- 余弦退火
- 动态调整

## 2. 损失函数
- 交叉熵损失
- 标签平滑(0.1)
- 自适应权重

## 3. 训练技巧
- 梯度裁剪
- 早停机制
- 混合精度训练

---

# 5. 创新亮点

## 数据处理
- 多领域融合
- 严格清洗
- 科学划分

## 模型设计
- 残差增强
- 多层特征
- 有效正则化

## 训练优化
- 自适应学习
- 防过拟合
- 混合精度

---

# 评估指标

![评估指标 width:800px](https://www.plantuml.com/plantuml/png/SoWkIImgAStDuNBCoKnELT2rKt3AJx9IS2mjoKZDAybCJYp9pCzJ24ejB4qjBk42oYde0000)

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC

---

# 谢谢！

**联系方式**
- Email: your.email@example.com
- GitHub: github.com/yourusername 