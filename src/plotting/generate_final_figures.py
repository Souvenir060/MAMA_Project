#!/usr/bin/env python3
"""
MAMA Framework Final Figure Generation
Academic Implementation - ONLY uses experimental results
STRICTLY PROHIBITS any hardcoded, simulated, or fake data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

def load_results() -> Dict[str, Any]:
    """
    Load experimental results from files
    INTEGRITY: Only uses results from actual experiments
    """
    results_dir = Path('results')
    
    # Check for real results file
    results_file = results_dir / 'final_results.json'
    if not results_file.exists():
        logger.error("❌ No experimental results found!")
        logger.error("   Results file must exist: results/final_results.json")
        logger.error("   Run experiments first: python run_experiments.py --mode core")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"✅ Loaded experimental results from {results_file}")
        
        # Validate that results contain actual experimental data
        required_keys = ['statistics', 'report', 'timestamp']
        for key in required_keys:
            if key not in results:
                logger.error(f"❌ Missing required key in results: {key}")
                return None
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to load results: {e}")
        return None

def generate_main_performance_figure(results: Dict[str, Any]):
    """
    Generate main performance comparison from REAL experimental data ONLY
    ACADEMIC INTEGRITY: No simulated or hardcoded data allowed
    """
    if not results:
        logger.error("❌ No results available for figure generation!")
        logger.error("   Cannot generate without experimental data!")
        return False
    
    # Extract statistics
    stats = results.get('statistics', [])
    if not stats:
        logger.error("❌ No statistics found in results!")
        return False
    
    # Extract model performance
    models = []
    mrr_means = []
    mrr_stds = []
    ndcg_means = []
    ndcg_stds = []
    
    for stat in stats:
        models.append(stat.get('model', 'Unknown'))
        mrr_means.append(stat.get('MRR_mean', 0.0))
        mrr_stds.append(stat.get('MRR_std', 0.0))
        ndcg_means.append(stat.get('NDCG@5_mean', 0.0))
        ndcg_stds.append(stat.get('NDCG@5_std', 0.0))
    
    # Create performance comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MRR comparison
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars1 = ax1.bar(models, mrr_means, yerr=mrr_stds, color=colors, 
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Mean Reciprocal Rank (MRR)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, mrr_means, mrr_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # NDCG@5 comparison  
    bars2 = ax2.bar(models, ndcg_means, yerr=ndcg_stds, color=colors,
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Normalized Discounted Cumulative Gain@5', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Normalized Discounted Cumulative Gain@5', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars2, ndcg_means, ndcg_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for both subplots
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    Path('figures').mkdir(exist_ok=True)
    
    # Save figure
    plt.savefig('figures/main_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/main_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Generated from experimental data")
    return True

def generate_all_figures():
    """
    Generate all figures from experimental results ONLY
    ACADEMIC INTEGRITY: Completely prohibits fake or simulated data
    """
    logger.info("🎯 Starting figure generation from experimental results")
    
    # Load experimental results
    results = load_results()
    if not results:
        logger.error("❌ ACADEMIC INTEGRITY VIOLATION PREVENTED!")
        logger.error("   Cannot generate without experimental data!")
        logger.error("   Run experiments first: python run_experiments.py --mode core")
        return False
    
    logger.info("🔬 ACADEMIC INTEGRITY VERIFIED: Using only real experimental data")
    
    # Generate main performance figure
    success = generate_main_performance_figure(results)
    
    if success:
        logger.info("✅ All figures generated successfully from REAL data")
        logger.info("📊 Integrity maintained: No simulated data used")
        return True
    else:
        logger.error("❌ Figure generation failed")
        return False

def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("MAMA Framework Final Figure Generation")
    logger.info("ACADEMIC INTEGRITY: Only experimental results allowed")
    logger.info("=" * 80)
    
    success = generate_all_figures()
    
    if not success:
        logger.error("❌ Figure generation failed - Run experiments first!")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
