#!/usr/bin/env python3
"""
MAMA Main Figures Generator
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime
import os
import sys
from typing import Dict, Any

def load_experimental_results(parent_dir):
    """
    Load experimental results from final_results.json
    
    Returns:
        Dictionary containing experimental results
    """
    final_results_path = Path(parent_dir) / "results" / "final_results.json"
    
    if not final_results_path.exists():
        print("❌ ERROR: No experimental results found!")
        print(f"   Expected file: {final_results_path}")
        print("\n🔧 To generate results, run:")
        print("   python run_experiments.py --mode core --verbose")
        print("   OR")
        print("   cd src && python evaluation/run_main_evaluation.py --verbose")
        sys.exit(1)
    
    try:
        with open(final_results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print(f"✅ Loaded experimental results from: {final_results_path}")
        return results_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load experimental results: {e}")
        sys.exit(1)

def load_hyperparameter_results(parent_dir):
    """Load hyperparameter sensitivity results if available"""
    hyperparam_path = Path(parent_dir) / "results" / "hyperparameter_sensitivity_results.json"
    
    if hyperparam_path.exists():
        try:
            with open(hyperparam_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load hyperparameter results: {e}")
    
    return None

def load_competence_results(parent_dir):
    """Load agent competence evolution results if available"""
    competence_path = Path(parent_dir) / "results" / "agent_competence_evolution_results.json"
    
    if competence_path.exists():
        try:
            with open(competence_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load competence results: {e}")
    
    return None

def extract_model_statistics(results_data: dict) -> Dict[str, Any]:
    """
    Extract model statistics from experimental results
    
    Args:
        results_data: Experimental results dictionary
        
    Returns:
        Dictionary with model statistics organized by model name
    """
    performance_stats = results_data.get('performance_statistics', [])
    
    if not performance_stats:
        print("❌ No performance statistics found in results")
        return {}
    
    # Organize by model for easier access
    model_data = {}
    
    for stat in performance_stats:
        model_name = stat.get('model', 'Unknown')
        
        # Map to consistent display names
        if model_name == 'MAMA Full':
            display_name = 'MAMA_Full'
        elif model_name == 'MAMA No Trust':
            display_name = 'MAMA_NoTrust' 
        elif model_name == 'Single Agent':
            display_name = 'Single_Agent'
        elif model_name == 'Traditional':
            display_name = 'Traditional'
        else:
            display_name = model_name.replace(' ', '_')
        
        model_data[display_name] = {
            'MRR': stat.get('MRR_mean', 0.0),
            'MRR_std': stat.get('MRR_std', 0.0),
            'NDCG@5': stat.get('NDCG@5_mean', 0.0),
            'NDCG@5_std': stat.get('NDCG@5_std', 0.0),
            'ART': stat.get('Response_Time_mean', 0.0),
            'ART_std': stat.get('Response_Time_std', 0.0)
        }
    
    print(f"📊 Found performance data for {len(model_data)} models")
    return model_data

def main():
    """Main execution - generates figures from REAL experimental data only"""
    print("🚀 Generating MAMA Framework Figures from Real Experimental Data")
    print("="*80)
    
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(parent_dir)
    
    # Load experimental results (will exit if not found)
    results_data = load_experimental_results(parent_dir)
    performance_stats = extract_model_statistics(results_data)
    
    # Load additional experimental results
    hyperparam_results = load_hyperparameter_results(parent_dir)
    competence_results = load_competence_results(parent_dir)
    
    # Create figures directory
    figures_dir = Path(parent_dir) / "figures"
    basic_dir = figures_dir / "basic"
    extended_dir = figures_dir / "extended"
    
    basic_dir.mkdir(exist_ok=True)
    extended_dir.mkdir(exist_ok=True)
    
    # Print experimental metadata
    metadata = results_data.get('metadata', {})
    print(f"📅 Experiment Date: {results_data.get('evaluation_date', 'Unknown')}")
    print(f"📋 Total Queries: {results_data.get('total_queries', 'Unknown')}")
    print(f"⏱️  Duration: {metadata.get('evaluation_duration_seconds', 0):.1f} seconds")
    print(f"🎯 Random Seed: {metadata.get('random_seed', 'Unknown')}")
    print("="*80)
    
    # Generate figures using real data
    generate_figure_6(performance_stats, basic_dir)
    generate_figure_7(hyperparam_results, basic_dir)
    generate_figure_8(competence_results, basic_dir)
    
    print("="*80)
    print("✅ All figures generated successfully from real experimental data!")
    print(f"📂 Figures saved to: {basic_dir}")
    print("="*80)

def generate_figure_6(performance_stats: Dict[str, Any], output_dir: Path):
    """
    Generate Figure 6: Main Performance Comparison with blue-white color scheme
    """
    print("📊 Generating Figure 6: Main Performance Comparison")
    
    # Extract data
    models = list(performance_stats.keys())
    mrr_values = [performance_stats[model]['MRR'] for model in models]
    mrr_stds = [performance_stats[model]['MRR_std'] for model in models]
    
    ndcg5_values = [performance_stats[model]['NDCG@5'] for model in models]
    ndcg5_stds = [performance_stats[model]['NDCG@5_std'] for model in models]
    
    art_values = [performance_stats[model]['ART'] for model in models]
    
    # Calculate improvement percentages for annotations
    mama_full_mrr = mrr_values[0]  # MAMA Full is first
    traditional_mrr = mrr_values[-1]  # Traditional is last
    single_agent_mrr = mrr_values[2] if len(mrr_values) > 2 else mrr_values[1]
    
    improvement_vs_traditional = ((mama_full_mrr - traditional_mrr) / traditional_mrr) * 100
    improvement_vs_single = ((mama_full_mrr - single_agent_mrr) / single_agent_mrr) * 100
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MAMA System Performance Evaluation', fontsize=16, fontweight='bold')
    
    # 🎨 Blue-White Color Scheme (requested by user)
    colors = ['#1E3A8A', '#3B82F6', '#60A5FA', '#DBEAFE']  # Deep blue to light blue-white gradient
    
    # Plot 1: MRR (NO ERROR BARS)
    bars1 = ax1.bar(models, mrr_values, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    ax1.set_title('(a) Mean Reciprocal Rank (MRR)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('MRR', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, color='lightgray')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_facecolor('#FAFBFC')  # Light background
    
    # Add value annotations on bars
    for i, (bar, value) in enumerate(zip(bars1, mrr_values)):
        height = bar.get_height()
        ax1.annotate(f'{value:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=10, color='#1E3A8A')
    
    # Plot 2: NDCG@5 (NO ERROR BARS)
    bars2 = ax2.bar(models, ndcg5_values, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    ax2.set_title('(b) Normalized Discounted Cumulative Gain@5', fontweight='bold', fontsize=12)
    ax2.set_ylabel('NDCG@5', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, color='lightgray')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_facecolor('#FAFBFC')  # Light background
    
    # Add value annotations on bars
    for i, (bar, value) in enumerate(zip(bars2, ndcg5_values)):
        height = bar.get_height()
        ax2.annotate(f'{value:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=10, color='#1E3A8A')
    
    # Plot 3: ART (NO ERROR BARS)
    bars3 = ax3.bar(models, art_values, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    ax3.set_title('(c) Average Response Time', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_ylim(0, max(art_values) * 1.2)
    ax3.grid(True, alpha=0.3, color='lightgray')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_facecolor('#FAFBFC')  # Light background
    
    # Add value annotations only for non-zero values
    for i, (bar, value) in enumerate(zip(bars3, art_values)):
        height = bar.get_height()
        if value > 0.001:  # Only annotate values > 1ms
            ax3.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10, color='#1E3A8A')
        else:
            ax3.annotate('<0.001',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=9, color='#1E3A8A')
    
    # Plot 4: Performance Improvement over Traditional
    improvement_models = ['MAMA_Full', 'MAMA_NoTrust', 'Single_Agent']
    improvement_values = [
        improvement_vs_traditional,
        ((mrr_values[1] - traditional_mrr) / traditional_mrr) * 100,  # MAMA No Trust vs Traditional
        ((single_agent_mrr - traditional_mrr) / traditional_mrr) * 100  # Single Agent vs Traditional
    ]
    
    bars4 = ax4.bar(improvement_models, improvement_values, color=colors[:3], alpha=0.9, edgecolor='white', linewidth=2)
    ax4.set_title('(d) Performance Improvement over Traditional', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Improvement (%)', fontweight='bold')
    ax4.set_ylim(0, max(improvement_values) * 1.2)
    ax4.grid(True, alpha=0.3, color='lightgray')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_facecolor('#FAFBFC')  # Light background
    
    # Add value annotations
    for i, (bar, value) in enumerate(zip(bars4, improvement_values)):
        height = bar.get_height()
        ax4.annotate(f'{value:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=10, color='#1E3A8A')
    
    # Add improvement summary with blue-white styling
    ax4.text(0.5, 0.95, f'MAMA Full: +{improvement_vs_traditional:.1f}% vs Traditional\n+{improvement_vs_single:.1f}% vs Single Agent', 
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F2FE", edgecolor="#1E3A8A", alpha=0.9),
             fontsize=10, fontweight='bold', color='#1E3A8A')
    
    plt.tight_layout()
    
    # Save figure with white background
    output_file = output_dir / "figure_6_main_evaluation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    
    print(f"✅ Figure 6 saved to: {output_file}")
    print(f"   🎨 Blue-white color scheme applied, data annotations added, no error bars")
    print(f"   📊 MAMA Full improvements: +{improvement_vs_traditional:.1f}% vs Traditional, +{improvement_vs_single:.1f}% vs Single Agent")

def generate_figure_7(hyperparam_results, output_dir):
    """Generate Figure 7: Hyperparameter Sensitivity - matching exact user style with STRAIGHT LINES"""
    print("📊 Generating Figure 7: Hyperparameter Sensitivity")
    
    if hyperparam_results is None:
        print("⚠️ No hyperparameter sensitivity results found!")
        print("   Run: python src/experiments/main_experiments/hyperparameter_sensitivity_experiment.py")
        print("   Skipping Figure 7...")
        return
    
    # Extract sensitivity results
    sensitivity_data = hyperparam_results.get('sensitivity_results', [])
    if not sensitivity_data:
        print("⚠️ No sensitivity data found in hyperparameter results!")
        return
    
    # Organize data for plotting
    alpha_values = sorted(list(set([r['alpha'] for r in sensitivity_data])))
    beta_values = sorted(list(set([r['beta'] for r in sensitivity_data])))
    
    # Create alpha sensitivity curve
    alpha_mrr_map = {}
    for result in sensitivity_data:
        alpha = result['alpha']
        if alpha not in alpha_mrr_map:
            alpha_mrr_map[alpha] = []
        alpha_mrr_map[alpha].append(result['mrr_mean'])
    
    # Average MRR for each alpha value
    alphas = sorted(alpha_mrr_map.keys())
    mrr_means = [np.mean(alpha_mrr_map[alpha]) for alpha in alphas]
    
    # Create heatmap data
    heatmap_data = np.zeros((len(beta_values), len(alpha_values)))
    for result in sensitivity_data:
        alpha_idx = alpha_values.index(result['alpha'])
        beta_idx = beta_values.index(result['beta'])
        heatmap_data[beta_idx, alpha_idx] = result['mrr_mean']
    
    # Create figure with 2 subplots (matching user's style exactly)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # (a) MRR vs Alpha Parameter - 🎨 蓝色系配色 (论文风格)
    axes[0].plot(alphas, mrr_means, 'o-', linewidth=2.5, markersize=8, color='#1f4e79')
    axes[0].set_title('(a) MRR vs Alpha Parameter', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Alpha (α)', fontsize=12)
    axes[0].set_ylabel('MRR', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Set Y-axis range to better show MRR variation
    mrr_min = min(mrr_means) * 0.95  # Add some padding
    mrr_max = max(mrr_means) * 1.05
    axes[0].set_ylim(mrr_min, mrr_max)
    
    # Highlight optimal point with 🌟 star marker (use GLOBAL best, not alpha average best)
    best_config = hyperparam_results.get('best_configuration', {})
    if best_config and 'alpha' in best_config:
        # Find the best alpha in our alpha list and mark it
        try:
            best_alpha = best_config['alpha'] 
            if best_alpha in alphas:
                best_alpha_idx = alphas.index(best_alpha)
                axes[0].plot(best_alpha, mrr_means[best_alpha_idx], '*', markersize=15, 
                            color='white', markeredgecolor='#1f4e79', markeredgewidth=2)
            else:
                # Fallback to closest alpha value
                closest_idx = np.argmin([abs(a - best_alpha) for a in alphas])
                axes[0].plot(alphas[closest_idx], mrr_means[closest_idx], '*', markersize=15, 
                            color='white', markeredgecolor='#1f4e79', markeredgewidth=2)
        except (ValueError, KeyError):
            # Fallback to original behavior if best_config is invalid
            max_idx = np.argmax(mrr_means)
            axes[0].plot(alphas[max_idx], mrr_means[max_idx], '*', markersize=15, 
                        color='white', markeredgecolor='#1f4e79', markeredgewidth=2)
    else:
        # Fallback to original behavior if no best_config
        max_idx = np.argmax(mrr_means)
        axes[0].plot(alphas[max_idx], mrr_means[max_idx], '*', markersize=15, 
                    color='white', markeredgecolor='#1f4e79', markeredgewidth=2)
    
    # (b) MRR Heatmap (α, β) - 🎨 蓝色系热图 (专业学术风格)
    # Set color range based on actual data for better visualization
    vmin = np.min(heatmap_data[heatmap_data > 0])  # Exclude zeros
    vmax = np.max(heatmap_data)
    im = axes[1].imshow(heatmap_data, cmap='Blues', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('(b) MRR Heatmap (α, β)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Alpha (α)', fontsize=12)
    axes[1].set_ylabel('Beta (β)', fontsize=12)
    
    # Set tick labels
    x_ticks = np.arange(len(alpha_values))
    y_ticks = np.arange(len(beta_values))
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    axes[1].set_xticklabels([f'{alpha:.1f}' for alpha in alpha_values])
    axes[1].set_yticklabels([f'{beta:.1f}' for beta in beta_values])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('MRR', fontsize=12)
    
    # Mark optimal point
    best_config = hyperparam_results.get('best_configuration', {})
    if best_config:
        try:
            best_alpha_idx = alpha_values.index(best_config['alpha'])
            best_beta_idx = beta_values.index(best_config['beta'])
            axes[1].plot(best_alpha_idx, best_beta_idx, '*', markersize=15, 
                        color='white', markeredgecolor='black', markeredgewidth=2)
            axes[1].text(best_alpha_idx + 0.5, best_beta_idx, 
                        f'Optimal (α={best_config["alpha"]:.2f}, β={best_config["beta"]:.2f})', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                        fontsize=9, fontweight='bold')
        except (ValueError, KeyError):
            pass
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = output_dir / "figure_7_hyperparameter_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 7 saved to: {output_path}")
    
    plt.close()

def generate_figure_8(competence_results, output_dir):
    """Generate Figure 8: Agent Competence Evolution - matching exact user style"""
    print("📊 Generating Figure 8: Agent Competence Evolution Analysis")
    
    if competence_results is None:
        print("⚠️ No agent competence evolution results found!")
        print("   Run: python src/experiments/main_experiments/agent_competence_evolution_experiment.py")
        print("   Skipping Figure 8...")
        return
    
    # Extract system rewards and competence evolution data
    system_rewards = competence_results.get('system_rewards', [])
    competence_evolution = competence_results.get('competence_evolution', {})
    
    if not system_rewards or not competence_evolution:
        print("⚠️ No valid competence evolution data found!")
        return
    
    # Prepare data
    interactions = np.arange(1, len(system_rewards) + 1)
    
    # Calculate moving average for system rewards
    window_size = 10
    moving_avg = []
    for i in range(len(system_rewards)):
        start_idx = max(0, i - window_size + 1)
        avg = np.mean(system_rewards[start_idx:i+1])
        moving_avg.append(avg)
    
    # Agent names and colors (matching user's figure exactly)
    agent_display_names = {
        'weather_agent': 'Weather Agent',
        'safety_assessment_agent': 'Safety Agent', 
        'flight_info_agent': 'Flight Agent',
        'economic_agent': 'Economic Agent',
        'integration_agent': 'Integration Agent'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create figure with 2 subplots (matching user's style)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Agent Competence Evolution Analysis', fontsize=16, fontweight='bold')
    
    # (a) System Reward Evolution
    axes[0].plot(interactions, system_rewards, color='blue', alpha=0.7, linewidth=1, label='System Reward')
    axes[0].plot(interactions, moving_avg, color='red', linewidth=2, label='10-point Moving Average')
    axes[0].set_title('(a) System Reward Evolution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Interaction Number', fontsize=12)
    axes[0].set_ylabel('System Reward', fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # (b) Agent Competence Evolution
    for i, (agent_key, display_name) in enumerate(agent_display_names.items()):
        if agent_key in competence_evolution:
            scores = competence_evolution[agent_key]
            if len(scores) == len(interactions):
                axes[1].plot(interactions, scores, color=colors[i], linewidth=2, 
                           marker='o', markersize=2, label=display_name, alpha=0.8)
    
    axes[1].set_title('(b) Agent Competence Evolution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Interaction Number', fontsize=12)
    axes[1].set_ylabel('Competence Score', fontsize=12)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = output_dir / "figure_8_agent_competence_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 8 saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
