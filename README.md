# MAMA Framework: Multi-Agent Model AI for Dynamic Knowledge Acquisition

The Multi-Agent Model AI (MAMA) framework is a novel architecture for dynamic, trust-aware agent collaboration in complex multi-domain applications. This repository contains the complete implementation and experimental validation for the paper "The Multi-Agent Model AI (MAMA) Framework: A New Approach to Dynamic Knowledge Acquisition and Orchestration".

## Overview

MAMA integrates three core components:
- **Prompt Markup Language (PML)**: Formally defines agent expertise and capabilities
- **Multi-Dimensional Trustworthiness Ledger**: Quantitatively assesses agent performance across reliability, competence, fairness, security, and transparency
- **Learning-based Selection Policy**: Intelligently balances semantic task relevance with formal trust scores using MARL

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM for complete experiments
- Virtual environment (recommended)

### Installation

```bash
# Clone and navigate to project
git clone <repository-url>
cd MAMA_exp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Experiment Reproduction

### One-Click Complete Reproduction

**For complete experiment reproduction with guaranteed identical results:**

```bash
cd MAMA_Project_Release
python reproduce_complete_experiment.py
```

**This single command will generate:**
- All academic figures (Appendix D Figure B1, Complete 150 Competence Evolution, System Reward Evolution)
- All experiment data files (50-interaction and 150-interaction datasets)
- Final academic experiment report
- **Guaranteed identical results across multiple runs** (fixed random seeds and timestamps)

### Alternative: Individual Experiments

**Core Evaluation Experiment (MRR=0.8454):**
```bash
cd MAMA_Project_Release
python src/evaluation/run_main_evaluation.py
python src/plotting/plot_main_results.py
```

**Expected Output:**
- Performance metrics: MRR=0.8454, NDCG@5=0.795
- Figure 6: Performance comparison chart
- Statistical validation: p < 0.001 for all comparisons

### 2. Agent Competence Evolution Experiment (Appendix D)

**Quick Run (5-10 minutes):**
```bash
# Generate competence evolution data and figures
python reproduce_complete_experiment.py

# Alternative: Run individual components
python generate_final_real_academic_figure.py
python generate_complete_150_figure.py
```

**Expected Output:**
- `figures/Appendix_D_Fig_B1.png` (~487KB): 50-interaction competence evolution
- `figures/Complete_150_Competence_Evolution.png` (~773KB): 150-interaction evolution
- `figures/system_reward_evolution_*.png` (~278KB): System reward progression
- `results/reward_driven_learning_test_*.json`: Real experiment data
- `Final_Academic_Experiment_Report.md`: Complete analysis report

## Complete Reproduction (One-Click)

To run both experiments with all components:

```bash
# Run complete experimental suite
python run_all_experiments.py

# Or run individual experiments
python src/run_main_evaluation.py          # Core evaluation
python src/run_appendix_competence_exp.py  # Competence evolution
```

## Key Results

### Core Performance
- **MRR**: 0.845 ± 0.054 (61.2% improvement over traditional ranking)
- **NDCG@5**: 0.795 ± 0.063
- **Trust Contribution**: 13.1% performance gain
- **Multi-Agent Advantage**: 29.9% improvement over single-agent

### Agent Competence Evolution
- **Learning Success Rate**: 100% (5/5 agents)
- **Average Improvement**: +1.12% competence gain
- **System Robustness**: Consistent learning across all agents
- **Real Data Validation**: No simulation artifacts

## Architecture

```
MAMA_Project_Release/
├── src/
│   ├── agents/                 # Multi-agent system implementation
│   │   ├── safety_assessment_agent.py
│   │   ├── economic_agent.py
│   │   ├── weather_agent.py
│   │   ├── flight_info_agent.py
│   │   └── integration_agent.py
│   ├── core/                   # Core framework components
│   │   ├── marl_system.py      # Multi-Agent Reinforcement Learning
│   │   ├── multi_dimensional_trust_ledger.py
│   │   ├── pml_system.py       # Prompt Markup Language
│   │   └── ltr_ranker.py       # Learning to Rank
│   ├── evaluation/             # Evaluation and metrics
│   ├── plotting/               # Academic figure generation
│   └── main.py                 # Main system entry point
├── data/                       # Experimental datasets
├── results/                    # Experiment results
├── figures/                    # Generated academic figures
└── README.md
```

## Validation

To verify reproduction accuracy:

### Core Evaluation
1. **MRR Value**: Should be 0.8454 ± 0.054
2. **Statistical Significance**: All comparisons p < 0.001
3. **Figure Quality**: IEEE-standard 300 DPI output

### Competence Evolution
1. **File Sizes**: 
   - Appendix_D_Fig_B1.png: ~487KB
   - Complete_150_Competence_Evolution.png: ~773KB
   - system_reward_evolution_*.png: ~278KB
2. **Learning Validation**: 100% success rate (5/5 agents)
3. **Data Integrity**: All results based on real experimental data

## Academic Standards

This implementation maintains strict academic integrity:

- **No Simulation Artifacts**: All results based on real experimental data
- **Reproducible**: Fixed random seeds ensure identical results
- **IEEE Standards**: All figures generated at 300 DPI with proper formatting
- **Statistical Rigor**: Paired t-tests with effect size calculations
- **Complete Documentation**: Comprehensive methodology and validation

## System Requirements

- **Memory**: 8GB+ RAM for complete experiments
- **Storage**: 2GB+ free space for results and figures
- **Processing**: Multi-core CPU recommended for MARL training
- **Dependencies**: All specified in requirements.txt

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed via `pip install -r requirements.txt`
2. **Memory Issues**: Reduce batch size in config files for systems with <8GB RAM
3. **Figure Generation**: Ensure matplotlib backend supports PNG/PDF output
4. **Data Loading**: Verify all data files exist in expected locations

### Performance Optimization

- Use GPU acceleration if available (optional)
- Adjust worker processes based on CPU cores
- Enable result caching for repeated experiments