# MAMA Framework Sentiment Analysis Case Study

## Overview

This is a sentiment analysis case study for the paper appendix, demonstrating the generalizability of the MAMA (Multi-Agent Multi-API) framework to natural language processing tasks.

## Experimental Design

### Task Description
- **Dataset**: Stanford Sentiment Treebank (SST-2)
- **Task**: Binary sentiment classification (0: Negative, 1: Positive)
- **Evaluation Set**: Validation set (872 samples)

### Agent Architecture
1. **Positive Agent**: Specializes in identifying positive sentiment expressions
2. **Negative Agent**: Specializes in identifying negative sentiment expressions
3. **Sarcasm Agent**: Specializes in detecting sarcasm and irony
4. **Negation Agent**: Specializes in identifying negation structures
5. **Aggregator Agent**: Rule-based knowledge fusion system

### MAMA Workflow
1. **PML Definition**: Define expertise domain descriptions for each agent
2. **Agent Selection**: Select Top-3 relevant experts based on semantic similarity
3. **Parallel Execution**: Selected agents independently analyze input sentences
4. **Knowledge Fusion**: Rule-based aggregation strategy to generate final predictions

## File Description

### Main Scripts
- `sentiment_analysis_case_study.py`: Complete experiment script (offline version)
- `sentiment_requirements.txt`: Python dependency list

### Usage

```bash
# 1. Install dependencies
pip install -r sentiment_requirements.txt

# 2. Run the experiment
python sentiment_analysis_case_study.py
```

## Experimental Results Format

### Console Output
After completion, the experiment will display in the console:
- Accuracy comparison between methods
- Markdown table for the paper appendix
- Detailed experiment configuration information

### Results File
- `sentiment_case_study_results_[timestamp].json`: Detailed experimental data

### Paper Table Example
```markdown
| Method | Accuracy | Samples |
|--------|----------|---------|
| MAMA Framework | 0.8234 (82.34%) | 50 |
| Single Agent Baseline | 0.7456 (74.56%) | 50 |
| **Improvement** | **+7.78 percentage points** | - |
```

## Technical Details

### Key Features
- **Minimal Modifications**: Reuses existing MAMA framework architecture
- **Explainability**: Every decision step is traceable
- **Robustness**: Includes error handling and retry mechanisms
- **Extensibility**: Easy to add new expert agents

### Aggregation Rules
1. **Sarcasm Priority**: If sarcasm is detected, directly classified as negative
2. **Negation Reversal**: If negation is present, reverse positive/negative votes
3. **Majority Voting**: In other cases, use simple majority voting

### Semantic Selection
- Uses `all-MiniLM-L6-v2` to compute semantic similarity between sentences and PML
- Selects Top-3 most relevant experts based on cosine similarity
- Supports dynamic agent combinations, enhancing task adaptability

## Citation Format

If using this case study in a paper, the suggested description format is:

> To verify the generalizability of the MAMA framework, we conducted a case study on the Stanford Sentiment Treebank (SST-2) dataset for sentiment analysis. The experiment involved four specialized sentiment analysis agents, dynamically selected through semantic similarity, with a rule-based aggregation strategy for knowledge fusion. Results showed that the MAMA framework improved accuracy by X.XX percentage points over the single-agent baseline, confirming the framework's effectiveness and generalizability in NLP tasks. 