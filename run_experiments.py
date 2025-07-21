#!/usr/bin/env python3
"""
MAMA Framework -  Unified Experiment Runner

This script provides a centralized interface for running all MAMA experiments.

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
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_runner.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner(text):
    """Print formatted banner"""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}\n")
    logger.info(f"Starting: {text}")

def create_directories():
    """Create necessary directories"""
    dirs = [
        'results', 
        'figures/basic', 
        'figures/extended',
        'src/results',
        'src/figures',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")
    return True

def run_script(script_path, description, args=None):
    """Run script and handle errors"""
    logger.info(f"Running: {description} - {script_path}")
    print(f"🚀 {description}")
    print(f"   Running: {script_path}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=str(Path(__file__).parent)
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            logger.info(f"Success: {description}")
            
            # Show only key output lines
            output_lines = result.stdout.strip().split('\n')
            key_lines = [line for line in output_lines if any(keyword in line for keyword in 
                        ['✓', '✅', '📊', '📁', '📝', 'Generated', 'MRR', 'NDCG', 'Success'])]
            
            for line in key_lines[:10]:  # Limit output lines
                print(f"   {line}")
                
            if len(key_lines) > 10:
                print(f"   ... and {len(key_lines) - 10} more results")
        else:
            print(f"❌ Error in {description}")
            print(f"   Error: {result.stderr}")
            logger.error(f"Error in {description}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        logger.error(f"Exception in {description}: {e}")
        return False
    
    return True

def check_environment():
    """Check environment dependencies"""
    print_banner("Environment Check")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check key dependencies
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        import sentence_transformers
        
        print("✅ Core dependencies available")
        logger.info("Core dependencies check passed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        logger.error(f"Dependency check failed: {e}")
        return False

def run_core_experiments(verbose=False, use_cache=True):
    """Run core MAMA framework experiments"""
    print_banner("Core Experiments")
    
    # Parameters
    args = []
    if not use_cache:
        args.append("--no-cache")
    if verbose:
        args.append("--verbose")
        
    # Run main evaluation (replaces final_150_test_experiment.py)
    success = run_script(
        "src/run_main_evaluation.py",
        "Main MAMA Framework Evaluation (150 test queries)",
        args
    )
    
    if not success:
        logger.error("Core experiment failed")
        return False
    
    # Generate primary result figures
    success = run_script(
        "src/plotting/plot_main_figures.py",
        "Generate Main Result Figures",
        args
    )
        
    print("\n✅ Core experiments completed successfully")
    logger.info("Core experiments completed successfully")
    return True

def run_robustness_experiments(verbose=False, use_cache=True):
    """Run robustness analysis experiments"""
    print_banner("Robustness Analysis Experiments")
    
    args = []
    if not use_cache:
        args.append("--no-cache")
    if verbose:
        args.append("--verbose")
    
    # Run ground truth robustness experiment
    success = run_script(
        "src/experiments/main_experiments/ground_truth_robustness_experiment.py",
        "Ground Truth Robustness Analysis",
        args
    )
    
    if not success:
        logger.error("Robustness experiment failed")
        return False
    
    # Run appendix robustness experiment
    success = run_script(
        "src/run_appendix_robustness.py",
        "Appendix: Extended Robustness Analysis",
        args
    )
    
    print("\n✅ Robustness experiments completed successfully")
    logger.info("Robustness experiments completed successfully")
    return True

def run_agent_competence_evolution(verbose=False, use_cache=True):
    """Run agent competence evolution experiments"""
    print_banner("Agent Competence Evolution Experiments")
    
    args = []
    if not use_cache:
        args.append("--no-cache")
    if verbose:
        args.append("--verbose")
    
    # Run competence evolution experiment
    success = run_script(
        "src/run_final_competence_experiment.py",
        "Agent Competence Evolution Analysis",
        args
    )
    
    if not success:
        logger.error("Competence evolution experiment failed")
        return False
    
    # Run appendix competence experiment
    success = run_script(
        "src/run_appendix_competence_exp.py",
        "Appendix: Extended Competence Analysis",
        args
    )
    
    print("\n✅ Competence evolution experiments completed successfully")
    logger.info("Competence evolution experiments completed successfully")
    return True

def run_all_figures(verbose=False):
    """Generate all figures from existing results"""
    print_banner("Generate All Figures")
    
    args = ["--regenerate"] if verbose else []
    
    # Main figures
    success = run_script(
        "src/plotting/plot_main_figures.py",
        "Main Evaluation Figures",
        args
    )
    
    # Appendix competence figures
    success &= run_script(
        "src/plotting/plot_appendix_competence_150.py",
        "Appendix: Competence Evolution (150 interactions)",
        args
    )
    
    # Appendix competence figures (50 interactions)
    success &= run_script(
        "src/plotting/plot_appendix_competence_50.py",
        "Appendix: Competence Evolution (50 interactions)",
        args
    )
    
    if not success:
        logger.error("Figure generation failed")
        return False
    
    print("\n✅ All figures generated successfully")
    logger.info("All figures generated successfully")
    return True

def run_appendix_experiments(verbose=False, use_cache=True):
    """Run appendix experiments"""
    print_banner("Appendix Experiments")
    
    args = []
    if not use_cache:
        args.append("--no-cache")
    if verbose:
        args.append("--verbose")
    
    # Appendix: Few-shot learning
    run_script(
        "src/run_appendix_fewshot.py",
        "Appendix: Few-Shot Learning Experiment",
        args
    )
    
    # Appendix: Reward-driven adaptation
    run_script(
        "src/run_appendix_reward_driven.py",
        "Appendix: Reward-Driven Adaptation Experiment",
        args
    )
    
    # Appendix: Scalability analysis
    run_script(
        "src/run_appendix_scalability.py",
        "Appendix: Scalability Analysis",
        args
    )
    
    print("\n✅ Appendix experiments completed successfully")
    logger.info("Appendix experiments completed successfully")
    return True

def generate_academic_report():
    """Generate comprehensive academic report"""
    print_banner("Academic Report Generation")
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    report_file = Path(f'results/MAMA_Academic_Report_{timestamp}.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MAMA Framework Academic Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Core results section
        f.write("## Core Results\n\n")
        try:
            result_files = list(Path('results').glob('final_run_150_test_set_*.json'))
            if result_files:
                latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
                with open(latest_result, 'r', encoding='utf-8') as rf:
                    results = json.load(rf)
                
                f.write("### Performance Metrics\n\n")
                f.write("| Model | MRR | NDCG@5 | Response Time (s) |\n")
                f.write("|-------|-----|--------|------------------|\n")
                
                for stat in results.get('statistics', []):
                    f.write(f"| {stat.get('model', 'N/A')} | ")
                    f.write(f"{stat.get('MRR_mean', 0):.4f}±{stat.get('MRR_std', 0):.4f} | ")
                    f.write(f"{stat.get('NDCG@5_mean', 0):.4f}±{stat.get('NDCG@5_std', 0):.4f} | ")
                    f.write(f"{stat.get('Response_Time_mean', 0):.3f}±{stat.get('Response_Time_std', 0):.3f} |\n")
                
                f.write("\n### Key Findings\n\n")
                for finding in results.get('report', {}).get('key_findings', []):
                    f.write(f"- {finding}\n")
        except Exception as e:
            f.write(f"Error generating core results section: {e}\n\n")
        
        # Robustness analysis section
        f.write("\n## Robustness Analysis\n\n")
        try:
            robustness_files = list(Path('results').glob('ground_truth_robustness_*.json'))
            if robustness_files:
                latest_robust = max(robustness_files, key=lambda p: p.stat().st_mtime)
                with open(latest_robust, 'r', encoding='utf-8') as rf:
                    robust_results = json.load(rf)
                
                f.write("| Filter Mode | Safety Threshold | Budget Multiplier | MAMA MRR | Single Agent MRR | Advantage (%) |\n")
                f.write("|-------------|-----------------|-------------------|----------|------------------|-------------|\n")
                
                for mode, data in robust_results.get('mode_results', {}).items():
                    f.write(f"| {mode} | {data.get('safety_threshold', 'N/A')} | ")
                    f.write(f"{data.get('budget_multiplier', 'N/A')}x | ")
                    f.write(f"{data.get('mama_full_mrr', 0):.4f} | ")
                    f.write(f"{data.get('single_agent_mrr', 0):.4f} | ")
                    f.write(f"{data.get('relative_advantage', 0):.2f}% |\n")
        except Exception as e:
            f.write(f"Error generating robustness section: {e}\n\n")
        
        # Competence evolution section
        f.write("\n## Agent Competence Evolution\n\n")
        f.write("See the competence evolution plots in the figures directory.\n\n")
        
    print(f"📄 Academic report generated: {report_file}")
    logger.info(f"Academic report generated: {report_file}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MAMA Framework Experiment Runner')
    parser.add_argument('--mode', type=str, default='core', 
                        choices=['core', 'full', 'figures', 'competence', 'robustness'],
                        help='Experiment mode')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force regeneration of results without using cache')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed output during execution')
    
    args = parser.parse_args()
    mode = args.mode
    verbose = args.verbose
    use_cache = not args.no_cache
    
    print_banner(f"MAMA Framework Experiment Runner - Mode: {mode.upper()}")
    logger.info(f"Starting experiment run in {mode} mode")
    
    # Setup environment
    if not check_environment():
        print("❌ Environment check failed")
        return 1
    
    if not create_directories():
        print("❌ Failed to create necessary directories")
        return 1
    
    # Execute experiments based on mode
    success = True
    
    if mode == 'core':
        success = run_core_experiments(verbose, use_cache)
    
    elif mode == 'figures':
        success = run_all_figures(verbose)
    
    elif mode == 'competence':
        success = run_agent_competence_evolution(verbose, use_cache)
    
    elif mode == 'robustness':
        success = run_robustness_experiments(verbose, use_cache)
    
    elif mode == 'full':
        # Run all experiments in sequence
        success = run_core_experiments(verbose, use_cache)
        if success:
            success &= run_robustness_experiments(verbose, use_cache)
        if success:
            success &= run_agent_competence_evolution(verbose, use_cache)
        if success:
            success &= run_appendix_experiments(verbose, use_cache)
        if success:
            success &= run_all_figures(verbose)
    
    # Generate final report
    if success:
        generate_academic_report()
        print_banner("MAMA Framework Experiment Completion")
        print("✅ All experiments completed successfully!")
        print("📁 Check the 'results/' directory for detailed outputs")
        print("📊 Check the 'figures/' directory for visualization")
        logger.info("All experiments completed successfully")
    else:
        print_banner("MAMA Framework Experiment Failed")
        print("❌ Some experiments failed. Check the logs for details.")
        logger.error("Experiment run failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 