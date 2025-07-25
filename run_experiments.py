#!/usr/bin/env python3
"""
MAMA Framework Comprehensive Experiment Runner
Orchestrates all experiments including core evaluation, robustness analysis, 
hyperparameter sensitivity, and agent competence evolution
"""

import argparse
import logging
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
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

def run_script(script_path: str, description: str, args: dict = None) -> bool:
    """
    Run a script with arguments
    
    Args:
        script_path: Path to the script
        description: Description for logging
        args: Dictionary of arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Running: {description} - {script_path}")
        print(f"🚀 {description}")
        print(f"   Running: {script_path}")
        
        # Build command
        cmd = [sys.executable, script_path]
        
        # Add arguments if provided
        if args:
            for key, value in args.items():
                if key == "verbose" and value:
                    cmd.append("--verbose")
                elif key == "use_cache" and not value:
                    cmd.append("--no-cache")
                elif key == "interactions":
                    cmd.extend(["--interactions", str(value)])
                elif key in ["data_path", "queries_path"] and value:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
                elif isinstance(value, bool) and value and key not in ["verbose", "use_cache"]:
                    cmd.append(f"--{key.replace('_', '-')}")
                elif not isinstance(value, bool) and key not in ["use_cache", "verbose", "interactions"]:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Execute script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            logger.info(f"Success: {description}")
            if result.stdout.strip():
                print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Error in {description}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            logger.error(f"Error in {description}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout in {description}")
        logger.error(f"Timeout in {description}")
        return False
    except Exception as e:
        print(f"💥 Exception in {description}: {e}")
        logger.error(f"Exception in {description}: {e}")
        return False

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

def run_core_experiments(verbose: bool, use_cache: bool) -> bool:
    """Run core experiments - Full Academic Standard"""
    logger.info("Starting: Core Experiments")
    
    success = True
    
    # Main MAMA Framework Evaluation - Full Academic Standard
    script_args = {"verbose": verbose, "use_cache": use_cache}
    
    success &= run_script(
        "src/evaluation/run_main_evaluation.py",  # Updated path
        "Main MAMA Framework Evaluation (150 test queries) - Full Academic Standard",
        script_args
    )
    
    return success

def run_full_experiments(verbose: bool, use_cache: bool) -> bool:
    """Run full experimental suite - Full Academic Standard"""
    logger.info("Starting: Full Experimental Suite - Full Academic Standard")
    
    success = True
    
    # Core experiments
    success &= run_core_experiments(verbose, use_cache)
    
    # Hyperparameter sensitivity experiment
    if success:
        hyper_args = {"verbose": verbose}
        
        success &= run_script(
            "src/experiments/main_experiments/hyperparameter_sensitivity_experiment.py",
            "Hyperparameter Sensitivity Analysis - Full Academic Standard",
            hyper_args
        )
        
    # Agent competence evolution experiment
    if success:
        comp_args = {"verbose": verbose, "interactions": 150}
        
        success &= run_script(
            "src/experiments/main_experiments/agent_competence_evolution_experiment.py",
            "Agent Competence Evolution Analysis - Full Academic Standard",
            comp_args
        )
    
    # Robustness experiments
    if success:
        success &= run_robustness_experiments(verbose, use_cache)
    
    return success

def run_robustness_experiments(verbose: bool, use_cache: bool) -> bool:
    """Run robustness analysis experiments"""
    logger.info("Starting: Robustness Analysis")
    
    success = True
    
    # Ground truth robustness
    success &= run_script(
        "src/experiments/main_experiments/robustness_analysis_final.py",
        "Ground Truth Robustness Analysis",
        {"verbose": verbose}
    )
    
    return success

def run_agent_competence_evolution(verbose: bool, use_cache: bool) -> bool:
    """Run agent competence evolution experiment - Full Academic Standard"""
    logger.info("Starting: Agent Competence Evolution Experiment - Full Academic Standard")
    
    comp_args = {"verbose": verbose, "interactions": 150}
    
    success = run_script(
        "src/experiments/main_experiments/agent_competence_evolution_experiment.py",
        "Agent Competence Evolution (150 interactions) - Full Academic Standard",
        comp_args
    )
    
    return success

def run_hyperparameter_sensitivity(verbose: bool) -> bool:
    """Run hyperparameter sensitivity experiment - Full Academic Standard"""
    logger.info("Starting: Hyperparameter Sensitivity Experiment - Full Academic Standard")
    
    hyper_args = {"verbose": verbose}
    
    success = run_script(
        "src/experiments/main_experiments/hyperparameter_sensitivity_experiment.py",
        "Hyperparameter Sensitivity Analysis - Full Academic Standard",
        hyper_args
    )
    
    return success

def run_appendix_experiments(verbose: bool, use_cache: bool) -> bool:
    """Run appendix experiments"""
    logger.info("Starting: Appendix Experiments")
    
    success = True
    
    # Hyperparameter sensitivity 
    success &= run_hyperparameter_sensitivity(verbose)
    
    # Additional robustness tests
    success &= run_script(
        "src/experiments/main_experiments/robustness_analysis.py",
        "Additional Robustness Analysis",
        {"verbose": verbose}
    )
    
    return success

def run_all_figures(verbose=False):
    """Generate all figures from existing results"""
    logger.info("Starting: Generate All Figures")
    
    # Main figures
    success = run_script(
        "src/plotting/plot_main_figures.py",
        "Main Evaluation Figures",
        {"verbose": verbose}
    )
    
    # Note: Appendix competence figures are now included in plot_main_figures.py as Figure 8
    
    if not success:
        logger.error("Figure generation failed")
        return False
    
    print("\n✅ All figures generated successfully")
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

def cleanup_project_files():
    """Clean up project files before running experiments"""
    print("🧹 Cleaning up project files...")
    
    # Directories to clean
    cleanup_dirs = [
        "results",
        "figures", 
        "src/results",
        "cache",
        "logs"
    ]
    
    # Clean directories
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                # Remove all files in directory but keep the directory
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        # Preserve hyperparameter sensitivity results
                        preserve_files = [
                            '.gitkeep',
                            'hyperparameter_sensitivity_results.json',
                            'hyperparameter_sensitivity_results_'  # Preserve timestamped versions too
                        ]
                        
                        should_preserve = any(
                            file.startswith(preserve_pattern) for preserve_pattern in preserve_files
                        )
                        
                        if not should_preserve:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        else:
                            print(f"    🏆 PRESERVED: {file} (hyperparameter results)")
                
                print(f"  ✅ Cleaned: {dir_path}/ (preserved hyperparameter results)")
            except Exception as e:
                print(f"  ⚠️ Could not clean {dir_path}: {e}")
    
    # Clean Python cache files
    try:
        import subprocess
        subprocess.run(["find", ".", "-name", "*.pyc", "-delete"], 
                      capture_output=True, check=False)
        subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], 
                      capture_output=True, check=False)
        print(f"  ✅ Cleaned: Python cache files")
    except:
        pass
    
    # Clean model checkpoints
    model_dirs = [
        "src/models/checkpoints",
        "src/core/marl_policy",
        "models"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            try:
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(('.pkl', '.json', '.pt', '.pth', '.ckpt')):
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                print(f"  ✅ Cleaned: {model_dir}/ checkpoints")
            except Exception as e:
                print(f"  ⚠️ Could not clean {model_dir}: {e}")
    
    print("✅ Project cleanup completed")
    print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MAMA Framework Experiment Runner')
    parser.add_argument('--mode', type=str, default='core', 
                        choices=['core', 'full', 'figures', 'competence', 'robustness', 'hyperparameter'],
                        help='Experiment mode')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force regeneration of results without using cache')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed output during execution')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip project cleanup before experiments')
    
    args = parser.parse_args()
    mode = args.mode
    verbose = args.verbose
    use_cache = not args.no_cache
    
    # Clean up project files unless explicitly disabled
    if not args.no_cleanup:
        cleanup_project_files()
    
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
    
    elif mode == 'hyperparameter':
        success = run_hyperparameter_sensitivity(verbose)
    
    elif mode == 'full':
        # Run all experiments in sequence - Full Academic Standard
        success = run_full_experiments(verbose, use_cache)
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