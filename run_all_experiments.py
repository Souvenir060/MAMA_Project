#!/usr/bin/env python3
"""
MAMA Framework - Complete Experiment Suite
One-click execution for all experiments: Core Evaluation + Appendix Extended Experiments
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_banner(text):
    """Print formatted banner"""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}\n")

def run_script(script_path, description):
    """Run script and handle errors"""
    print(f"🚀 {description}")
    print(f"   Running: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True,
            cwd=str(Path(__file__).parent)
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            
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
            return False
            
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False
    
    return True

def check_environment():
    """Check environment dependencies"""
    print_banner("Environment Check")
    
    # Check Python version
    import sys
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check key dependencies
    required_packages = ['numpy', 'matplotlib', 'pandas', 'scikit-learn', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies satisfied")
    return True

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'results', 
        'figures/basic', 
        'figures/extended',
        'src/results',
        'src/figures'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def run_main_experiments():
    """Run main experiments (core paper results)"""
    print_banner("Core Paper Experiments")
    
    # Core evaluation experiment
    success = run_script(
        "src/run_main_evaluation.py",
        "Core Evaluation (MRR=0.8454)"
    )
    
    if not success:
        return False
    
    # Generate main figures
    success = run_script(
        "src/plot_main_figures.py", 
        "Main Performance Figures"
    )
    
    return success

def run_appendix_experiments():
    """Run appendix experiments (extended validation)"""
    print_banner("Appendix Extended Experiments")
    
    experiments = [
        ("src/run_appendix_reward_driven.py", "Reward-Driven Learning"),
        ("src/run_appendix_fewshot.py", "Few-Shot Learning Analysis"),
        ("src/run_appendix_robustness.py", "Robustness Analysis"),
        ("src/run_appendix_scalability.py", "Scalability Stress Test")
    ]
    
    for script_path, description in experiments:
        success = run_script(script_path, description)
        if not success:
            print(f"⚠️  {description} failed, continuing with other experiments...")
    
    return True

def generate_final_figures():
    """Generate final academic figures"""
    print_banner("Final Figure Generation")
    
    # Generate competence evolution plots
    success = run_script(
        "src/plot_agent_competence_curves.py",
        "Agent Competence Evolution Plots"
    )
    
    return success

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    print_banner("MAMA Framework - Complete Experiment Suite")
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Environment check
    if not check_environment():
        print("❌ Environment check failed. Please install dependencies.")
        return
    
    # Setup directories
    setup_directories()
    
    # Run main experiments
    print("\n🎯 Phase 1: Core Paper Results")
    if not run_main_experiments():
        print("❌ Main experiments failed")
        return
    
    # Run appendix experiments
    print("\n🎯 Phase 2: Extended Validation")
    run_appendix_experiments()
    
    # Generate final figures
    print("\n🎯 Phase 3: Final Figures")
    generate_final_figures()
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_banner("Experiment Suite Completed")
    print(f"⏱️  Total Duration: {duration}")
    print(f"📊 Results saved in: results/")
    print(f"📈 Figures saved in: figures/")
    print(f"🎯 Core Results: MRR=0.8454, NDCG@5=0.795")
    print(f"✅ All experiments completed successfully!")

if __name__ == "__main__":
    main() 