# CAD Framework for Aspect-Based Sentiment Quadruplet Prediction

## 论文信息

**标题**: Bridging Consistency and Diversity Gaps in Aspect-Based Sentiment Quadruplet Prediction via Diffusion Self-Augmented

---

## 任务介绍

### ASQP任务

ASQP (Aspect Sentiment Quad Prediction) 是ABSA (Aspect-Based Sentiment Analysis) 中最具挑战性的任务，旨在从评论文本中识别方面级别的情感四元组。

每个四元组包含四个元素：
- **Aspect Term (at)**: 方面项/评价对象
- **Aspect Category (ac)**: 方面类别（预定义类别）
- **Opinion Term (ot)**: 观点项/情感表达词
- **Sentiment Polarity (sp)**: 情感倾向 (positive/negative/neutral)

**示例**:
```
输入: "the food is great and reasonably priced"
输出: {(food, food_quality, great, positive), (food, food_prices, reasonably priced, positive)}
```

---

## 代码结构

```
CAD for Github/
├── README.md                    # 本文档
├── requirements.txt             # 环境依赖
├── src/
│   ├── train_cad.py             # CAD框架主训练脚本
│   ├── train_quad.py            # 四元组预测基线训练
│   ├── model.py                 # 主模型 (CADModule)
│   ├── LightingModule_SL.py     # PyTorch Lightning模块
│   ├── modules/
│   │   ├── __init__.py          # 模块导出
│   │   ├── cg_module.py         # CG模块 - 伪数据生成
│   │   ├── made_module.py       # MADE模块 - 样本打分与筛选
│   │   ├── diffusion.py         # 高斯扩散模型
│   │   └── loss.py              # 损失函数
│   └── utils/
│       ├── __init__.py          # 工具导出
│       ├── quad.py              # 四元组解析
│       ├── quad_result.py       # 结果评估
│       ├── data_utils.py        # 数据处理
│       └── category_mappings.json
└── scripts/
    └── train_quad.sh            # 训练脚本
```

---

## 环境配置

### 硬件要求
- GPU: NVIDIA RTX 4090 (24GB)
- 内存: 64GB+

### 软件环境

```bash
# 创建conda环境
conda create -n st python=3.9
conda activate st

# 安装PyTorch (CUDA 11.8)
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install lightning==2.1.3
pip install transformers==4.37.0
pip install numpy==1.26.4
pip install scikit-learn==1.5.0
pip install spacy==3.7.4
pip install tqdm==4.67.3
pip install sentencepiece
pip install accelerate
```

### 预训练模型

下载T5-large模型到本地：
```bash
mkdir -p /path/to/model_base/t5_large
# 从HuggingFace下载 t5-large 到该目录
```

---

## 运行方法

### 1. 数据准备

数据集应放置在 `data/t5/` 目录下：
```
data/t5/
├── acos/
│   ├── laptop16/
│   └── rest16/
└── asqp/
    ├── rest15/
    └── rest16/
```

### 2. 训练流程

CAD框架的完整训练流程：

**Step 1**: 训练基线模型
```bash
cd src
python train_cad.py \
    --dataset acos/rest16 \
    --model_name_or_path /path/to/t5_large \
    --train_mode train_quad \
    --max_epochs 20 \
    --devices 0
```

**Step 2**: 使用CG模块生成伪数据样本
```bash
python train_cad.py \
    --dataset acos/rest16 \
    --train_mode pseudo_labeling \
    --use_cg True \
    --use_made False \
    --devices 0
```

**Step 3**: 使用MADE模块评估打分并筛选
```bash
python train_cad.py \
    --dataset acos/rest16 \
    --train_mode train_cad \
    --use_cg True \
    --use_made True \
    --devices 0
```

**Step 4**: 将筛选后的优质样本加入原始数据集继续训练

### 3. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | acos/rest16 | 数据集选择 |
| `--model_name_or_path` | t5_large | T5模型路径 |
| `--max_epochs` | 20 | 训练轮数 |
| `--train_batch_size` | 8 | 批量大小 |
| `--learning_rate` | 5e-5 | 学习率 |
| `--use_cg` | True | 是否使用CG模块（伪数据生成） |
| `--use_made` | True | 是否使用MADE模块（样本打分与筛选） |
| `--w_consistency` | 0.7 | 一致性权重 |
| `--w_diversity` | 0.3 | 多样性权重 |
| `--top_k` | 4 | 筛选top-k高质量样本 |

### 4. 自增强Pipeline使用示例

```python
from modules.made_module import SelfAugmentationPipeline, MADEEvaluator
from modules.cg_module import ControllableGeneration

# 初始化CG和MADE模块
cg = ControllableGeneration(hidden_dim=1024)
made = MADEEvaluator(hidden_dim=1024, w_consistency=0.7, w_diversity=0.3)

# 创建自增强Pipeline
pipeline = SelfAugmentationPipeline(cg, made, num_candidates=4)

# 对一个batch进行自增强
# 1. CG生成候选样本
# 2. MADE评估打分
# 3. 筛选高质量样本
augmented_samples, scores = pipeline.augment_dataset(batch, threshold=0.5)

# augmented_samples: 筛选后的高质量样本
# scores: 每个样本的评估分数 (diversity, consistency, final_score)
```

---

## 引用