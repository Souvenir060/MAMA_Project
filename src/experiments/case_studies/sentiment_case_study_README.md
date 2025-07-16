# MAMA框架情感分析案例研究

## 概述

这是一个用于论文附录的情感分析案例研究，展示了MAMA（Multi-Agent Multi-criteria Airline recommendation）框架在自然语言处理任务上的通用性。

## 实验设计

### 任务描述
- **数据集**: Stanford Sentiment Treebank (SST-2)
- **任务**: 二元情感分类 (0: Negative, 1: Positive)
- **评估集**: 验证集 (872个样本)

### 智能体架构
1. **Positive_Agent**: 专门识别积极情感表达
2. **Negative_Agent**: 专门识别消极情感表达  
3. **Sarcasm_Agent**: 专门检测讽刺和反讽
4. **Negation_Agent**: 专门识别否定结构
5. **Aggregator_Agent**: 规则基的知识融合器

### MAMA工作流程
1. **PML定义**: 为每个智能体定义专业领域描述
2. **智能体选择**: 基于语义相似度选择Top-3相关专家
3. **并行执行**: 选中的智能体独立分析输入句子
4. **知识融合**: 基于规则的聚合策略生成最终预测

## 文件说明

### 主要脚本
- `sentiment_analysis_case_study.py`: 完整的实验脚本（需要OpenAI API）
- `sentiment_demo_without_api.py`: 演示版本（不需要API，使用模拟智能体）
- `sentiment_requirements.txt`: Python依赖包列表

### 使用方法

#### 方法一：完整实验（需要API密钥）
```bash
# 1. 安装依赖
pip install -r sentiment_requirements.txt

# 2. 设置API密钥
export OPENAI_API_KEY="your-api-key-here"

# 3. 运行实验
python sentiment_analysis_case_study.py
```

#### 方法二：演示版本（不需要API）
```bash
# 1. 安装基础依赖
pip install datasets sentence-transformers scikit-learn numpy

# 2. 运行演示
python sentiment_demo_without_api.py
```

## 实验结果格式

### 控制台输出
实验完成后会在控制台显示：
- 各方法的准确率对比
- 论文附录用的Markdown表格
- 详细的实验配置信息

### 结果文件
- `sentiment_case_study_results_[timestamp].json`: 详细实验数据
- `sentiment_demo_results.json`: 演示结果（演示版本）

### 论文表格示例
```markdown
| 方法 | 准确率 | 样本数 |
|------|--------|--------|
| MAMA框架 | 0.8234 (82.34%) | 50 |
| 单智能体基线 | 0.7456 (74.56%) | 50 |
| **提升** | **+7.78 百分点** | - |
```

## 技术细节

### 关键特性
- **最小化改动**: 复用现有MAMA框架架构
- **可解释性**: 每个决策步骤都可追踪
- **鲁棒性**: 包含错误处理和重试机制
- **扩展性**: 容易添加新的专家智能体

### 聚合规则
1. **讽刺优先**: 如果检测到讽刺，直接判定为负面
2. **否定反转**: 如果有否定词，反转积极/消极投票
3. **多数投票**: 其他情况下采用简单多数投票

### 语义选择
- 使用 `all-mpnet-base-v2` 计算句子与PML的语义相似度
- 基于余弦相似度选择Top-3最相关的专家
- 支持动态智能体组合，提高任务适应性

## 注意事项

1. **API配置**: 完整版本需要有效的OpenAI API密钥
2. **成本控制**: 实验默认限制在50个样本以控制API调用成本
3. **延迟设置**: 包含请求延迟以避免API限制
4. **错误处理**: 包含完整的异常处理和默认值机制

## 引用格式

如果在论文中使用此案例研究，建议的描述格式：

> 为验证MAMA框架的通用性，我们在Stanford Sentiment Treebank (SST-2)数据集上进行了情感分析任务的案例研究。实验设计了四个专门的情感分析智能体，通过语义相似度进行动态选择，并采用规则基聚合策略进行知识融合。结果表明，MAMA框架相比单智能体基线在准确率上提升了X.XX个百分点，验证了框架在NLP任务上的有效性和通用性。 