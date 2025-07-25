#!/usr/bin/env python3
"""
MAMA Framework Installation Verification Script
Verifies that all core components are working correctly
"""

import json
import sys
from pathlib import Path

def check_files():
    """Check that all required files exist"""
    required_files = [
        'data/standard_dataset.json',
        'src/evaluation/run_main_evaluation.py',  # Updated path
        'src/plotting/plot_main_figures.py',
        'src/data/generate_standard_dataset.py',
        'requirements.txt',
        'run_experiments.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_dataset():
    """Verify dataset structure"""
    try:
        with open('data/standard_dataset.json', 'r') as f:
            data = json.load(f)
        
        train_size = len(data['train'])
        val_size = len(data['validation'])
        test_size = len(data['test'])
        
        if train_size == 700 and val_size == 150 and test_size == 150:
            print(f"✅ Dataset structure correct: {train_size} train, {val_size} val, {test_size} test")
            return True
        else:
            print(f"❌ Dataset structure incorrect: {train_size} train, {val_size} val, {test_size} test")
            return False
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def check_core_modules():
    """Verify core Python modules can be imported"""
    try:
        # Test core imports
        import src.models.mama_full
        import src.core.evaluation_metrics
        import src.data.generate_standard_dataset
        print("✅ Core modules import successfully")
                return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 MAMA Framework Installation Verification")
    print("=" * 50)
    
    checks = [
        ("File structure", check_files),
        ("Dataset integrity", check_dataset),
        ("Core modules", check_core_modules),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification checks passed!")
        print("✅ MAMA Framework is ready for experiments")
        print("\nNext steps:")
        print("1. Run: python run_experiments.py --mode core")
        print("2. Run: python src/plotting/plot_main_figures.py")
        print("3. Check figures/ directory for generated plots")
    else:
        print("❌ Some verification checks failed")
        print("Please fix the issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
