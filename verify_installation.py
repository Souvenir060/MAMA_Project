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
        'results/final_run_150_test_set_2025-07-04_18-03.json',
        'src/run_main_evaluation.py',
        'src/plot_main_figures.py',
        'src/generate_datasets.py',
        'requirements.txt'
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

def check_results():
    """Verify golden results file"""
    try:
        with open('results/final_run_150_test_set_2025-07-04_18-03.json', 'r') as f:
            data = json.load(f)
        
        # Find MAMA_Full results
        mama_stats = None
        for stat in data['performance_statistics']:
            if stat['model'] == 'MAMA_Full':
                mama_stats = stat
                break
        
        if mama_stats:
            mrr = mama_stats['MRR_mean']
            ndcg = mama_stats['NDCG@5_mean']
            
            if abs(mrr - 0.8454) < 0.001:
                print(f"✅ MAMA Full MRR verified: {mrr:.4f}")
                print(f"✅ MAMA Full NDCG@5: {ndcg:.4f}")
                return True
            else:
                print(f"❌ MAMA Full MRR mismatch: expected 0.8454, got {mrr:.4f}")
                return False
        else:
            print("❌ MAMA_Full results not found")
            return False
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔍 MAMA Framework Installation Verification")
    print("=" * 50)
    
    checks = [
        ("File structure", check_files),
        ("Dataset integrity", check_dataset),
        ("Results verification", check_results),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification checks passed!")
        print("✅ MAMA Framework is ready for use")
        print("\nNext steps:")
        print("1. Run: python src/plot_main_figures.py")
        print("2. Check figures/ directory for generated plots")
        print("3. Verify Figure 6 matches paper results")
    else:
        print("❌ Some verification checks failed")
        print("Please fix the issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
