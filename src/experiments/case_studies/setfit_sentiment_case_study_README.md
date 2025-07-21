# MAMA Framework: SetFit Sentiment Analysis Case Study

## Overview

This is a few-shot sentiment analysis case study using SetFit for the paper appendix, demonstrating the MAMA framework's effectiveness in few-shot learning scenarios.

## Experimental Design

### Task Description
- **Dataset**: Stanford Sentiment Treebank (SST-2)
- **Task**: Binary sentiment classification (0: Negative, 1: Positive)
- **Training**: Few-shot learning with only 8 samples per class
- **Evaluation Set**: Validation set (872 samples)

### Agent Architecture
1. **SetFit Agent**: Specialized in few-shot text classification using Sentence Transformers
2. **Template Agent**: Uses fixed templates and rules for sentiment classification
3. **Meta Learning Agent**: Implements meta-learning for few-shot adaptation
4. **MAMA Integration Agent**: Orchestrates the multi-agent system

### MAMA Workflow
1. **Few-Shot Training**: Train agents on minimal labeled data (8 samples per class)
2. **Agent Selection**: Select the most appropriate agent based on the input query
3. **Adaptive Fusion**: Combine predictions using trust-weighted voting
4. **Meta-Learning Adaptation**: Continuously improve through meta-learning

## File Description

### Main Scripts
- `setfit_sentiment_case_study.py`: Complete experiment script
- `sentiment_requirements.txt`: Python dependency list (includes SetFit)

### Usage

```bash
# 1. Install dependencies
pip install -r sentiment_requirements.txt

# 2. Run the experiment
python setfit_sentiment_case_study.py
```

## Experimental Results Format

### Console Output
After completion, the experiment will display in the console:
- Few-shot learning accuracy comparison
- Markdown table for the paper appendix
- Detailed experiment configuration information

### Results File
- `setfit_sentiment_case_study_results_[timestamp].json`: Detailed experimental data

### Paper Table Example
```markdown
| Method | Accuracy | Training Samples | Test Samples |
|--------|----------|------------------|-------------|
| MAMA + SetFit | 0.7294 | 16 (8 per class) | 872 |
| SetFit Baseline | 0.6965 | 16 (8 per class) | 872 |
| **Improvement** | **+3.29 pp** | - | - |
```

## Technical Details

### Key Features
- **Few-Shot Learning**: Achieves good performance with minimal labeled data
- **Sentence Transformers**: Uses contrastive learning approach for efficient training
- **Integration with MAMA**: Demonstrates how MAMA framework enhances other ML approaches
- **Zero API Dependencies**: Runs entirely locally for reproducibility

### SetFit Integration
- SetFit uses a two-stage training approach:
  1. Fine-tune sentence embeddings using contrastive learning
  2. Train a classification head on the fine-tuned embeddings
- MAMA framework enhances this by:
  1. Selecting optimal model configurations
  2. Combining predictions from multiple SetFit variants
  3. Implementing adaptive weighting based on confidence

## Citation Format

If using this case study in a paper, the suggested description format is:

> To evaluate the MAMA framework in few-shot learning scenarios, we conducted a case study on sentiment analysis using the Stanford Sentiment Treebank (SST-2) dataset with only 8 samples per class. By integrating SetFit with the MAMA architecture, we observed a 3.29 percentage point accuracy improvement over the standalone SetFit baseline, demonstrating the framework's ability to enhance performance even in data-scarce scenarios. 