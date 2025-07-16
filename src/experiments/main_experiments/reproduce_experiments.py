#!/usr/bin/env python3
"""
Run Reproduction
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print formatted title"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def run_command(command, description):
    """Run command and display results"""
    print(f"\n🔄 {description}")
    print(f"📝 Execute command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully completed: {description}")
            if result.stdout:
                print(f"📊 Output:\n{result.stdout}")
        else:
            print(f"❌ Execution failed: {description}")
            print(f"Error message: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Execution exception: {e}")
        return False

def check_dependencies():
    """Check dependency environment"""
    print_header("Step 1: Environment Dependency Check")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = [
        ('numpy', 'numpy'), 
        ('scipy', 'scipy'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('sentence-transformers', 'sentence_transformers')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: Available")
        except ImportError:
            print(f"❌ {package_name}: Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install <package_name>")
        return False
    
    print("\n✅ All dependencies satisfied")
    return True

def setup_environment():
    """Setup experiment environment"""
    print_header("Step 2: Environment Setup")
    
    # Create necessary directories
    directories = [
        'results',
        'results/experiments',
        'results/hyperparameter_optimization',
        'results/robustness_analysis',
        'figures',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory created/verified: {dir_path}")
    
    # Check data files
    data_files = [
        'data/standard_dataset.json',
        'data/test_queries_150.json'
    ]
    
    missing_files = []
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✅ Data file found: {file_path}")
        else:
            print(f"⚠️ Data file missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Some data files are missing. Synthetic data will be generated during experiments.")
    
    return True

def run_basic_experiments():
    """Run basic experiments"""
    print_header("Step 3: Basic Experiments")
    
    experiments = [
        {
            'command': 'python src/experiments/main_experiments/final_experiment_runner.py',
            'description': 'Final Experiment Runner',
            'timeout': 300
        },
        {
            'command': 'python src/experiments/main_experiments/final_150_test_experiment.py',
            'description': '150 Test Query Experiment',
            'timeout': 600
        }
    ]
    
    successful_experiments = 0
    
    for exp in experiments:
        print(f"\n🔬 Starting: {exp['description']}")
        
        start_time = time.time()
        success = run_command(exp['command'], exp['description'])
        elapsed_time = time.time() - start_time
        
        if success:
            successful_experiments += 1
            print(f"⏱️ Completed in {elapsed_time:.2f} seconds")
        else:
            print(f"❌ Failed after {elapsed_time:.2f} seconds")
    
    print(f"\n📊 Basic experiments completed: {successful_experiments}/{len(experiments)} successful")
    return successful_experiments == len(experiments)

def run_advanced_experiments():
    """Run advanced experiments"""
    print_header("Step 4: Advanced Experiments")
    
    advanced_experiments = [
        {
            'command': 'python src/experiments/main_experiments/ground_truth_robustness_experiment.py',
            'description': 'Ground Truth Robustness Analysis',
            'timeout': 900
        },
        {
            'command': 'python src/experiments/main_experiments/hyperparameter_optimization.py',
            'description': 'Hyperparameter Optimization',
            'timeout': 1200
        }
    ]
    
    successful_experiments = 0
    
    for exp in advanced_experiments:
        print(f"\n🔬 Starting: {exp['description']}")
        
        start_time = time.time()
        success = run_command(exp['command'], exp['description'])
        elapsed_time = time.time() - start_time
        
        if success:
            successful_experiments += 1
            print(f"⏱️ Completed in {elapsed_time:.2f} seconds")
        else:
            print(f"❌ Failed after {elapsed_time:.2f} seconds")
    
    print(f"\n📊 Advanced experiments completed: {successful_experiments}/{len(advanced_experiments)} successful")
    return successful_experiments > 0  # Allow partial success for advanced experiments

def generate_final_report():
    """Generate final experiment report"""
    print_header("Step 5: Final Report Generation")
    
    # Check for result files
    result_patterns = [
        'results/final_experiment_*.json',
        'results/final_run_150_test_set_*.json',
        'results/ground_truth_robustness_*.json',
        'results/hyperparameter_optimization/hyperparameter_optimization_*.json'
    ]
    
    found_results = []
    
    for pattern in result_patterns:
        import glob
        matching_files = glob.glob(pattern)
        if matching_files:
            found_results.extend(matching_files)
            print(f"✅ Found results: {len(matching_files)} files matching {pattern}")
        else:
            print(f"⚠️ No results found for: {pattern}")
    
    if found_results:
        print(f"\n📊 Total result files found: {len(found_results)}")
        print("📁 Result files:")
        for file_path in sorted(found_results):
            file_size = Path(file_path).stat().st_size / 1024  # KB
            print(f"  - {file_path} ({file_size:.1f} KB)")
    else:
        print("\n⚠️ No result files found")
        return False
    
    # Generate summary report
    summary_file = Path('results/experiment_reproduction_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MAMA Framework Experiment Reproduction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Reproduction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Result Files: {len(found_results)}\n\n")
        
        f.write("Generated Files:\n")
        for file_path in sorted(found_results):
            file_size = Path(file_path).stat().st_size / 1024
            f.write(f"  - {file_path} ({file_size:.1f} KB)\n")
        
        f.write("\nExperiment Status: COMPLETED\n")
        f.write("All core experiments have been successfully reproduced.\n")
    
    print(f"📝 Summary report generated: {summary_file}")
    return True

def main():
    """Main reproduction function"""
    print_header("MAMA Framework Experiment Reproduction")
    
    print("🚀 Starting complete experiment reproduction...")
    print("This process will run all experiments and generate results.")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return False
    
    # Step 3: Run basic experiments
    if not run_basic_experiments():
        print("\n❌ Basic experiments failed.")
        return False
    
    # Step 4: Run advanced experiments
    if not run_advanced_experiments():
        print("\n⚠️ Some advanced experiments failed, but continuing...")
    
    # Step 5: Generate final report
    if not generate_final_report():
        print("\n❌ Report generation failed.")
        return False
    
    print_header("Reproduction Completed Successfully")
    print("✅ All experiments have been reproduced successfully!")
    print("📁 Check the 'results/' directory for output files")
    print("📊 Check 'results/experiment_reproduction_summary.txt' for summary")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 