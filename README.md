# MAMA Framework: The Multi-Agent Model AI Framework

MAMA introduces a novel architecture designed to address the challenges of dynamic orchestration and collaborative trustworthiness in multi-agent AI systems. The framework enables dynamic, trust-aware agent collaboration by integrating a formal definition of agent expertise, a quantitative trust ledger, and a learning-based selection policy.

## Project Structure

```
MAMA_Project/
├── assets/                  # Visual assets and diagrams
├── data/                    # Dataset files 
│   └── standard_dataset.json  # Standard dataset for experiments (700/150/150 split)
│   └── test_queries.json    # 150-query test dataset
├── figures/                 # Output figures from experiments
├── models/                  # Model implementations
│   ├── base_model.py        # Base model class
│   ├── mama_full.py         # Complete MAMA model with trust
│   ├── mama_no_trust.py     # Ablation model without trust
│   ├── single_agent_system.py  # Single agent baseline
│   └── traditional_ranking.py  # Traditional ranking baseline
├── results/                 # Experimental results 
├── src/
│   ├── agents/              # Specialized agent implementations
│   │   ├── base_agent.py    # Base agent class
│   │   ├── flight_info_agent.py  # Flight information agent
│   │   ├── safety_assessment_agent.py  # Safety assessment agent
│   │   ├── economic_agent.py  # Economic agent
│   │   ├── integration_agent.py  # Integration agent
│   │   ├── manager.py       # Agent orchestration
│   │   ├── trust_manager.py  # Trust management system
│   │   └── weather_agent.py  # Weather agent
│   ├── core/                # Core framework components
│   │   ├── csv_flight_data.py  # Flight data processor
│   │   ├── ltr_ranker.py    # Learning to Rank implementation
│   │   ├── marl_environment.py  # MARL environment
│   │   ├── marl_policy.py   # MARL policy implementation
│   │   ├── marl_system.py   # MARL system integration
│   │   ├── multi_dimensional_trust_ledger.py  # Trust ledger implementation
│   │   └── sbert_similarity.py  # SBERT similarity engine
│   ├── experiments/         # Experiment implementations
│   │   ├── case_studies/    # Additional case studies
│   │   │   └── sentiment_analysis_case_study.py  # Sentiment analysis case study
│   │   └── main_experiments/  # Main paper experiments
│   │       ├── experiment_1_comparative_analysis.py  # Model comparison
│   │       ├── hyperparameter_optimization.py  # Alpha/Beta optimization
│   │       └── ground_truth_robustness_experiment.py  # Robustness testing
│   ├── evaluation/          # Evaluation metrics and tools
│   │   └── standard_evaluator.py  # Standard evaluation metrics (MRR, NDCG)
│   └── plotting/            # Figure generation
│       ├── plot_main_figures.py  # Generate Figure 6 (main comparison)
│       └── plot_appendix_competence_150.py  # Figure 8 (competence evolution)
├── flights.csv              # Flight dataset (336,777 records)
├── README.md                # This file
├── run_experiments.py       # Unified experiment runner
├── requirements.txt         # Python dependencies
├── sentiment_requirements.txt  # Dependencies for sentiment case study
└── verify_installation.py   # Installation verification script
```

## Core Components

### Agent System
The MAMA framework consists of 5 specialized agents that work collaboratively:

- **FlightInfoAgent**: Retrieves and processes flight data from the flights.csv dataset
- **WeatherAgent**: Analyzes weather conditions for departure and arrival cities
- **SafetyAssessmentAgent**: Evaluates flight safety based on airline, aircraft, and route
- **EconomicAgent**: Analyzes flight costs and economic factors
- **IntegrationAgent**: Integrates results from all agents to produce final rankings

### Technical Components

- **Prompt Markup Language (PML)**: Formal agent expertise definition
- **Multi-Dimensional Trust Ledger**: 5-dimensional trust assessment with weighted computation:
  - Reliability (25%): Consistency in task completion
  - Competence (25%): Quality and accuracy of outputs
  - Fairness (15%): Absence of bias in decisions
  - Security (15%): Resilience against attacks
  - Transparency (20%): Clarity of decision-making
- **SBERT Similarity Engine**: Using all-MiniLM-L6-v2 model for semantic matching
- **MARL System**: DQN-based reinforcement learning for agent selection
- **LTR Engine**: LambdaMART algorithm for result ranking

## Installation

```bash
# Clone the repository
git clone https://github.com/Souvenir060/MAMA_Project.git
cd MAMA_Project

# Install main dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

## Running Experiments

### Unified Experiment Runner

The `run_experiments.py` script serves as the single unified entry point for all MAMA experiments. It provides flexible options to run different types of experiments:

```bash
# Run core experiments (default)
python run_experiments.py --mode core

# Run the complete experiment suite including all appendix studies
python run_experiments.py --mode full

# Generate only figures from existing results
python run_experiments.py --mode figures

# Run agent competence evolution experiments
python run_experiments.py --mode competence

# Run robustness analysis experiments
python run_experiments.py --mode robustness
```

#### Command-line Options

```
Usage:
    python run_experiments.py [--mode {core,full,figures,competence,robustness}]
                             [--no-cache] [--verbose]

Options:
    --mode core         Run only core experiments (default)
    --mode full         Run all experiments including appendix studies
    --mode figures      Generate only figures from existing results
    --mode competence   Run agent competence evolution experiments
    --mode robustness   Run robustness analysis experiments
    --no-cache          Force regeneration of results without using cache
    --verbose           Display detailed output during execution
```

The unified runner automatically:
- Creates all required directories
- Manages dependencies between experiments 
- Ensures proper execution order
- Generates comprehensive reports and figures
- Logs all activities for traceability

### Mode Descriptions

1. **Core Mode (`--mode core`)**:
   - Runs main model comparison on 150 test queries
   - Evaluates MAMA Full, MAMA No Trust, Single Agent, and Traditional
   - Generates key performance metrics (MRR, NDCG@5)
   - Produces main result figures

2. **Full Mode (`--mode full`)**:
   - Runs everything in Core mode
   - Adds all appendix experiments
   - Includes robustness analysis and competence evolution
   - Generates all figures and a comprehensive academic report

3. **Figures Mode (`--mode figures`)**:
   - Generates all figures from existing results without running experiments
   - Includes main comparison charts and competence evolution plots
   - Useful for regenerating visuals after experiments are complete

4. **Competence Mode (`--mode competence`)**:
   - Runs only agent competence evolution experiments
   - Tracks how agent trust scores evolve over 50 and 150 interactions
   - Generates competence evolution figures

5. **Robustness Mode (`--mode robustness`)**:
   - Runs ground truth sensitivity analysis
   - Tests system with different parameter settings
   - Validates that MAMA's performance advantage is consistent

### Specialized Scripts

These specialized scripts can still be used individually for specific tasks:

```bash
# Generate visualization figures
python src/plotting/plot_main_figures.py

# Run competence evolution analysis
python src/run_final_competence_experiment.py
```

### Sentiment Analysis Case Study

```bash
# Install additional dependencies
pip install -r sentiment_requirements.txt

# Run the sentiment case study
python src/experiments/case_studies/sentiment_analysis_case_study.py
```

### Sentiment Analysis Case Study (Table II)

```bash
# Install additional dependencies for sentiment analysis
pip install datasets

# Run complete sentiment analysis experiment (872 samples from SST-2)
cd src/experiments/case_studies
python sentiment_analysis_case_study.py

# Results will be saved to results/sentiment_case_study_results_*.json
# Expected: MAMA ~49%, Single Agent Baseline ~53% (domain mismatch expected)
```

## Dataset

The project uses a comprehensive flight dataset (`flights.csv`, 41MB) containing 336,777 real flight records. The dataset is processed by `src/core/csv_flight_data.py` and split into:
- 700 training queries
- 150 validation queries
- 150 test queries

## Reproduction Instructions

### Prerequisites

1. **Python Environment**: Python 3.8+ with conda/pip
2. **System Requirements**: 8GB+ RAM, 2GB+ disk space
3. **Internet Connection**: Required for SBERT model downloads

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd MAMA_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python verify_installation.py
```

### Reproducing Paper Results

#### Method 1: Complete Automated Reproduction (Recommended)

```bash
# Run complete experiment pipeline (~20 minutes)
# This follows correct academic order: hyperparameters → main evaluation → evolution → figures
python run_experiments.py --mode full --verbose

# Check all results
ls results/
ls figures/basic/
```

#### Method 2: Step-by-Step Manual Reproduction

**Step 1: Find Optimal Hyperparameters** (Required First!)
```bash
# Run hyperparameter sensitivity analysis (63 combinations × 150 queries, ~18 minutes)
python src/experiments/main_experiments/hyperparameter_sensitivity_experiment.py

# Verify optimal parameters found
python -c "
from src.models.mama_full import MAMAFull
model = MAMAFull()
print(f'Optimal: α={model.config.alpha:.2f}, β={model.config.beta:.2f}, γ={model.config.gamma:.2f}')
"
```

**Step 2: Run Main Evaluation with Optimal Parameters**
```bash
# Run main model comparison (150 test queries, ~1 minute)
python src/evaluation/run_main_evaluation.py

# Check core results
ls results/final_results.json
```

**Step 3: Run Agent Competence Evolution with Optimal Parameters**
```bash
# Run agent evolution analysis (150 interactions, ~1 minute)
python src/experiments/main_experiments/agent_competence_evolution_experiment.py

# Check evolution results
ls results/agent_competence_evolution*
```

**Step 4: Generate All Academic Figures**
```bash
# Generate Figure 6 (Main Comparison), Figure 7 (Hyperparameters), Figure 8 (Evolution)
python src/plotting/plot_main_figures.py

# Verify all figures generated
ls figures/basic/
# Expected files: figure_6_main_evaluation.png, figure_7_hyperparameter_sensitivity.png, figure_8_agent_competence_evolution.png
# - figure_6_main_evaluation.png
# - figure_7_hyperparameter_sensitivity.png  
# - figure_8_agent_competence_evolution.png
```

### Validation and Verification

```bash
# 1. Verify real ground truth data is used
head -20 src/data/test_queries.json | grep "ground_truth_ranking"

# 2. Check experiment logs for real data processing
tail -50 experiment_runner.log | grep "ground truth"

# 3. Verify no hardcoded results
grep -r "hardcode\|fake\|simulate" src/ --exclude-dir=__pycache__
```

#### Common Issues

1. **SBERT Model Download Timeout**:
   ```bash
   # Solution: Use offline model cache
   export TRANSFORMERS_OFFLINE=1
   ```

2. **Memory Issues**:
   ```bash
   # Solution: Use turbo mode for faster execution
   python run_experiments.py --mode core --turbo
   ```

3. **Missing Results**:
   ```bash
   # Solution: Check experiment completion
   python run_experiments.py --mode core --verbose
   ```

## Key Files

### Core Framework
- **src/core/multi_dimensional_trust_ledger.py**: Implements the 5-dimensional trust system
- **src/core/marl_system.py**: Implements Multi-Agent Reinforcement Learning
- **src/core/sbert_similarity.py**: Implements semantic similarity matching
- **src/core/ltr_ranker.py**: Implements Learning to Rank algorithm

### Model Implementations
- **models/mama_full.py**: Complete MAMA implementation with trust mechanism
- **models/mama_no_trust.py**: Ablation model with trust mechanism disabled
- **models/single_agent_system.py**: Single agent baseline with true serial execution
- **models/traditional_ranking.py**: Traditional BM25+rule based baseline

### Evaluation
- **src/run_main_evaluation.py**: Main evaluation script for all models
- **src/evaluation/standard_evaluator.py**: Implements MRR and NDCG metrics

### Runner Scripts
- **run_experiments.py**: Unified experiment runner with multiple execution modes

## Reproducibility

All experiments use a fixed random seed (42) for reproducibility. The evaluation is conducted on a blind test set of 150 queries. Ground truth is generated using a non-compensatory lexicographic preference ordering model to ensure an unbiased assessment.

### Details

- **Dataset**: Complete SST-2 validation set (872 samples)
- **MAMA Implementation**: 
  - Uses SBERT for agent selection
  - Multi-agent framework with positive, negative, sarcasm, and negation detection agents
  - Weighted voting aggregation with precedence rules
- **Baseline**: Rule-based sentiment classifier with positive/negative word lists and negation handling
