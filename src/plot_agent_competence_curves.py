#!/usr/bin/env python3
"""
Script for drawing the growth curve of intelligent agent capability
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

# Set font configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_competence_data(log_file):
    """Load competence evolution data"""
    try:
        df = pd.read_csv(log_file)
        print(f"✅ Successfully loaded data file: {log_file}")
        print(f"📊 Data overview: {len(df)} records, {df['Interaction_ID'].nunique()} interactions")
        return df
    except Exception as e:
        print(f"❌ Failed to load data file: {e}")
        return None

def plot_competence_evolution(df, output_dir="figures", show_plot=True):
    """Plot agent competence evolution curves"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAMA Framework Agent Competence Evolution Analysis\nMulti-Agent Model AI Framework - Agent Competence Evolution', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color configuration
    colors = {
        'safety_agent_competence': '#FF6B6B',
        'economic_agent_competence': '#4ECDC4', 
        'weather_agent_competence': '#45B7D1',
        'flight_info_agent_competence': '#96CEB4',
        'integration_agent_competence': '#FFEAA7'
    }
    
    # Agent name mapping
    agent_names = {
        'safety_agent_competence': 'Safety Assessment Agent',
        'economic_agent_competence': 'Economic Analysis Agent',
        'weather_agent_competence': 'Weather Information Agent',
        'flight_info_agent_competence': 'Flight Information Agent',
        'integration_agent_competence': 'Integration Coordination Agent'
    }
    
    # Plot 1: Overall competence evolution
    ax1 = axes[0, 0]
    for col in colors.keys():
        if col in df.columns:
            ax1.plot(df['Interaction_ID'], df[col], 
                    color=colors[col], 
                    linewidth=2.5, 
                    label=agent_names[col],
                    marker='o', markersize=4, alpha=0.8)
            
    ax1.set_title('Agent Competence Evolution Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Interaction Number', fontsize=12)
    ax1.set_ylabel('Competence Score', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Competence growth rate
    ax2 = axes[0, 1]
    for col in colors.keys():
        if col in df.columns:
            # Calculate growth rate (difference between consecutive points)
            growth_rate = df[col].diff().fillna(0)
            ax2.plot(df['Interaction_ID'], growth_rate, 
                    color=colors[col], 
                    linewidth=2, 
                    label=agent_names[col],
                    alpha=0.7)
    
    ax2.set_title('Competence Growth Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Interaction Number', fontsize=12)
    ax2.set_ylabel('Growth Rate', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Cumulative competence improvement
    ax3 = axes[1, 0]
    for col in colors.keys():
        if col in df.columns:
            # Calculate cumulative improvement from initial value
            initial_value = df[col].iloc[0]
            cumulative_improvement = df[col] - initial_value
            ax3.plot(df['Interaction_ID'], cumulative_improvement, 
                    color=colors[col], 
                    linewidth=2.5, 
                    label=agent_names[col],
                    marker='s', markersize=3, alpha=0.8)
    
    ax3.set_title('Cumulative Competence Improvement', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Interaction Number', fontsize=12)
    ax3.set_ylabel('Cumulative Improvement', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Competence distribution (final values)
    ax4 = axes[1, 1]
    final_competences = []
    agent_labels = []
    
    for col in colors.keys():
        if col in df.columns:
            final_competences.append(df[col].iloc[-1])
            agent_labels.append(agent_names[col].replace(' Agent', ''))
    
    bars = ax4.bar(range(len(final_competences)), final_competences, 
                   color=[colors[col] for col in colors.keys() if col in df.columns],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax4.set_title('Final Competence Scores', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Agents', fontsize=12)
    ax4.set_ylabel('Final Competence Score', fontsize=12)
    ax4.set_xticks(range(len(agent_labels)))
    ax4.set_xticklabels(agent_labels, rotation=45, ha='right')
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_competences):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'agent_competence_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Figure saved to: {output_path}")
    
    # Save as PDF too
    pdf_path = Path(output_dir) / 'agent_competence_evolution.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"📄 PDF saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_learning_efficiency_analysis(df, output_dir="figures", show_plot=True):
    """Plot learning efficiency analysis"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAMA Framework Learning Efficiency Analysis\nAgent Learning Patterns and Convergence Behavior', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color configuration
    colors = {
        'safety_agent_competence': '#FF6B6B',
        'economic_agent_competence': '#4ECDC4', 
        'weather_agent_competence': '#45B7D1',
        'flight_info_agent_competence': '#96CEB4',
        'integration_agent_competence': '#FFEAA7'
    }
    
    # Agent name mapping
    agent_names = {
        'safety_agent_competence': 'Safety Assessment',
        'economic_agent_competence': 'Economic Analysis',
        'weather_agent_competence': 'Weather Information',
        'flight_info_agent_competence': 'Flight Information',
        'integration_agent_competence': 'Integration Coordination'
    }
    
    # Plot 1: Learning velocity (smoothed growth rate)
    ax1 = axes[0, 0]
    window_size = 5
    
    for col in colors.keys():
        if col in df.columns:
            # Calculate smoothed growth rate
            growth_rate = df[col].diff().fillna(0)
            smoothed_growth = growth_rate.rolling(window=window_size, center=True).mean()
            
            ax1.plot(df['Interaction_ID'], smoothed_growth, 
                    color=colors[col], 
                    linewidth=2.5, 
                    label=agent_names[col],
                    alpha=0.8)
    
    ax1.set_title('Learning Velocity (Smoothed Growth Rate)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Interaction Number', fontsize=12)
    ax1.set_ylabel('Learning Velocity', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Learning acceleration (second derivative)
    ax2 = axes[0, 1]
    
    for col in colors.keys():
        if col in df.columns:
            # Calculate second derivative (acceleration)
            first_diff = df[col].diff().fillna(0)
            second_diff = first_diff.diff().fillna(0)
            
            ax2.plot(df['Interaction_ID'], second_diff, 
                    color=colors[col], 
                    linewidth=2, 
                    label=agent_names[col],
                    alpha=0.7)
    
    ax2.set_title('Learning Acceleration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Interaction Number', fontsize=12)
    ax2.set_ylabel('Learning Acceleration', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Convergence analysis
    ax3 = axes[1, 0]
    
    for col in colors.keys():
        if col in df.columns:
            # Calculate distance from final value
            final_value = df[col].iloc[-1]
            distance_from_final = abs(df[col] - final_value)
            
            ax3.semilogy(df['Interaction_ID'], distance_from_final + 1e-6, 
                        color=colors[col], 
                        linewidth=2.5, 
                        label=agent_names[col],
                        alpha=0.8)
    
    ax3.set_title('Convergence Analysis (Distance from Final Value)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Interaction Number', fontsize=12)
    ax3.set_ylabel('Distance from Final Value (log scale)', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning efficiency comparison
    ax4 = axes[1, 1]
    
    # Calculate learning efficiency metrics
    efficiency_metrics = {}
    
    for col in colors.keys():
        if col in df.columns:
            # Total improvement
            total_improvement = df[col].iloc[-1] - df[col].iloc[0]
            
            # Average learning rate
            avg_learning_rate = total_improvement / len(df)
            
            # Stability (inverse of variance in growth rate)
            growth_rate = df[col].diff().fillna(0)
            stability = 1 / (growth_rate.var() + 1e-6)
            
            efficiency_metrics[agent_names[col]] = {
                'total_improvement': total_improvement,
                'avg_learning_rate': avg_learning_rate,
                'stability': stability
            }
    
    # Create efficiency score (weighted combination)
    efficiency_scores = []
    agent_labels = []
    
    for agent, metrics in efficiency_metrics.items():
        efficiency_score = (0.5 * metrics['total_improvement'] + 
                          0.3 * metrics['avg_learning_rate'] * 100 + 
                          0.2 * min(metrics['stability'], 1.0))
        efficiency_scores.append(efficiency_score)
        agent_labels.append(agent)
    
    bars = ax4.bar(range(len(efficiency_scores)), efficiency_scores, 
                   color=[colors[col] for col in colors.keys() if col in df.columns],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax4.set_title('Learning Efficiency Score', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Agents', fontsize=12)
    ax4.set_ylabel('Efficiency Score', fontsize=12)
    ax4.set_xticks(range(len(agent_labels)))
    ax4.set_xticklabels(agent_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'learning_efficiency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Learning efficiency analysis saved to: {output_path}")
    
    # Save as PDF too
    pdf_path = Path(output_dir) / 'learning_efficiency_analysis.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"📄 PDF saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_competence_statistics(df):
    """Generate competence evolution statistics"""
    
    print("\n📊 COMPETENCE EVOLUTION STATISTICS")
    print("=" * 50)
    
    # Agent columns
    agent_columns = [col for col in df.columns if 'competence' in col]
    
    # Agent name mapping
    agent_names = {
        'safety_agent_competence': 'Safety Assessment Agent',
        'economic_agent_competence': 'Economic Analysis Agent',
        'weather_agent_competence': 'Weather Information Agent',
        'flight_info_agent_competence': 'Flight Information Agent',
        'integration_agent_competence': 'Integration Coordination Agent'
    }
    
    for col in agent_columns:
        if col in df.columns:
            agent_name = agent_names.get(col, col)
            
            # Basic statistics
            initial_value = df[col].iloc[0]
            final_value = df[col].iloc[-1]
            max_value = df[col].max()
            min_value = df[col].min()
            
            # Improvement metrics
            total_improvement = final_value - initial_value
            improvement_percentage = (total_improvement / initial_value) * 100
            
            # Learning metrics
            growth_rate = df[col].diff().fillna(0)
            avg_growth_rate = growth_rate.mean()
            growth_variance = growth_rate.var()
            
            # Convergence metrics
            final_quarter = df[col].iloc[-len(df)//4:]
            convergence_stability = final_quarter.std()
            
            print(f"\n{agent_name}:")
            print(f"  Initial Competence: {initial_value:.4f}")
            print(f"  Final Competence: {final_value:.4f}")
            print(f"  Maximum Competence: {max_value:.4f}")
            print(f"  Minimum Competence: {min_value:.4f}")
            print(f"  Total Improvement: {total_improvement:.4f} ({improvement_percentage:+.1f}%)")
            print(f"  Average Growth Rate: {avg_growth_rate:.6f}")
            print(f"  Growth Variance: {growth_variance:.6f}")
            print(f"  Convergence Stability: {convergence_stability:.6f}")
    
    # Overall system statistics
    print(f"\n🎯 OVERALL SYSTEM PERFORMANCE:")
    
    # Calculate system-wide metrics
    system_initial = df[agent_columns].iloc[0].mean()
    system_final = df[agent_columns].iloc[-1].mean()
    system_improvement = system_final - system_initial
    system_improvement_pct = (system_improvement / system_initial) * 100
    
    print(f"  System Initial Average: {system_initial:.4f}")
    print(f"  System Final Average: {system_final:.4f}")
    print(f"  System Improvement: {system_improvement:.4f} ({system_improvement_pct:+.1f}%)")
    
    # Agent ranking by improvement
    improvements = {}
    for col in agent_columns:
        if col in df.columns:
            improvement = df[col].iloc[-1] - df[col].iloc[0]
            improvements[agent_names.get(col, col)] = improvement
    
    sorted_agents = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 AGENT RANKING BY IMPROVEMENT:")
    for i, (agent, improvement) in enumerate(sorted_agents, 1):
        print(f"  {i}. {agent}: {improvement:.4f}")
    
    print("=" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Plot agent competence evolution curves')
    parser.add_argument('--log_file', type=str, default='competence_evolution.log',
                       help='Path to competence evolution log file')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Load data
    df = load_competence_data(args.log_file)
    
    if df is None:
        print("❌ Cannot load data, exiting...")
        return
    
    # Generate plots
    print("📊 Generating competence evolution plots...")
    plot_competence_evolution(df, args.output_dir, args.show)
    
    print("📈 Generating learning efficiency analysis...")
    plot_learning_efficiency_analysis(df, args.output_dir, args.show)
    
    # Generate statistics
    generate_competence_statistics(df)
    
    print("\n✅ All plots and analysis completed!")

if __name__ == "__main__":
    main() 