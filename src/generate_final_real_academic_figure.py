#!/usr/bin/env python3
"""
Figure Generator
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from pathlib import Path
import seaborn as sns

# Set IEEE academic standard figure parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (7.5, 5.5)  # IEEE column width

class FinalFigureGenerator:
    """Real Academic Figure Generator - Based on Actual Experiment Data"""
    
    def __init__(self, results_dir="results"):
        """Initialize generator"""
        self.results_dir = Path(results_dir)
        
        # Load real reward-driven learning data
        self.data = None
        
        # IEEE academic standard colors (high distinguishability, suitable for B&W printing)
        self.colors = {
            'safety_assessment_agent': '#1f77b4',    # Blue
            'economic_agent': '#ff7f0e',             # Orange
            'weather_agent': '#2ca02c',              # Green
            'flight_info_agent': '#d62728',          # Red
            'integration_agent': '#9467bd'           # Purple
        }
        
        # English agent name mapping
        self.agent_names = {
            'safety_assessment_agent': 'Safety Assessment Agent',
            'economic_agent': 'Economic Agent',
            'weather_agent': 'Weather Agent',
            'flight_info_agent': 'Flight Info Agent',
            'integration_agent': 'Integration Agent'
        }
        
        # Create output directory
        os.makedirs("figures", exist_ok=True)
    
    def load_experiment_data(self):
        """Load real experiment data"""
        # Find latest result file
        result_files = list(self.results_dir.glob("reward_driven_learning_*.json"))
        if not result_files:
            raise FileNotFoundError("No experiment result files found")
        
        # Sort by date (newest first)
        latest_file = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        # Load data
        with open(latest_file, "r") as f:
            self.data = json.load(f)
        
        print(f"Loaded experiment data from: {latest_file}")
        return self.data
    
    def generate_competence_evolution_figure(self):
        """Generate competence evolution figure"""
        if self.data is None:
            self.load_experiment_data()
        
        # Create figure with IEEE proportions
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        
        # Plot each agent's competence evolution
        for agent_id in self.data["agent_ids"]:
            scores = self.data["competence_evolution"][agent_id]
            ax.plot(range(1, len(scores) + 1), scores, 
                   color=self.colors.get(agent_id, 'gray'), 
                   label=self.agent_names.get(agent_id, agent_id),
                   marker='o', markevery=5, markersize=5)
        
        # Set IEEE-compliant formatting
        ax.set_xlabel('Interaction Number', fontweight='bold')
        ax.set_ylabel('Agent Competence', fontweight='bold')
        ax.set_title('Agent Competence Evolution', fontweight='bold')
        
        # Add grid with specific styling
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Customize x-axis ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        # Add legend with IEEE styling
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)
        
        # Add figure number and reference text
        plt.figtext(0.01, 0.01, 'Figure 8: Agent competence evolution over 50 interactions.', 
                   ha='left', fontsize=9, fontstyle='italic')
        
        # Tight layout and save
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        # Save in multiple formats for academic publication
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(f"figures/competence_evolution_{timestamp}.{fmt}", 
                       bbox_inches='tight', format=fmt)
        
        print(f"Generated competence evolution figure: figures/competence_evolution_{timestamp}.png")
    
    def generate_reward_evolution_figure(self):
        """Generate system reward evolution figure"""
        if self.data is None:
            self.load_experiment_data()
        
        # Create figure with IEEE proportions
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        
        rewards = self.data["system_rewards"]
        interactions = range(1, len(rewards) + 1)
        
        # Plot system rewards
        ax.plot(interactions, rewards, color='#1f77b4', linestyle='-', 
               linewidth=1.5, alpha=0.7, label='System Reward')
        
        # Add 10-point moving average
        window_size = 10
        if len(rewards) >= window_size:
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                avg = np.mean(rewards[start_idx:end_idx])
                moving_avg.append(avg)
            
            ax.plot(interactions, moving_avg, color='#ff7f0e', linestyle='-', 
                   linewidth=2, label='10-Point Moving Average')
        
        # Set IEEE-compliant formatting
        ax.set_xlabel('Interaction Number', fontweight='bold')
        ax.set_ylabel('System Reward', fontweight='bold')
        ax.set_title('MAMA System Reward Evolution', fontweight='bold')
        
        # Add grid with specific styling
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Customize x-axis ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        
        # Add reference line at y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add legend with IEEE styling
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)
        
        # Add figure number and reference text
        plt.figtext(0.01, 0.01, 'Figure 7: System reward evolution over 50 interactions.', 
                   ha='left', fontsize=9, fontstyle='italic')
        
        # Tight layout and save
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        # Save in multiple formats for academic publication
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(f"figures/system_reward_evolution_{timestamp}.{fmt}", 
                       bbox_inches='tight', format=fmt)
        
        print(f"Generated system reward figure: figures/system_reward_evolution_{timestamp}.png")
    
    def generate_comparative_performance_figure(self):
        """Generate comparative performance figure"""
        # Model performance data (from paper)
        models = ['Traditional', 'Single\nAgent', 'MAMA\n(No Trust)', 'MAMA\n(Full)']
        mrr_scores = [0.525, 0.651, 0.747, 0.845]
        ndcg_scores = [0.481, 0.598, 0.688, 0.795]
        
        # Standard deviations (from paper)
        mrr_std = [0.053, 0.048, 0.051, 0.054]
        ndcg_std = [0.059, 0.055, 0.057, 0.063]
        
        # Create figure with IEEE proportions
        fig, ax = plt.subplots(figsize=(7.5, 5))
        
        # Set bar width and positions
        bar_width = 0.35
        x = np.arange(len(models))
        
        # Create grouped bars
        mrr_bars = ax.bar(x - bar_width/2, mrr_scores, bar_width, 
                         yerr=mrr_std, capsize=5,
                         label='MRR', color='#1f77b4', edgecolor='black', linewidth=1)
        
        ndcg_bars = ax.bar(x + bar_width/2, ndcg_scores, bar_width, 
                          yerr=ndcg_std, capsize=5,
                          label='NDCG@5', color='#ff7f0e', edgecolor='black', linewidth=1)
        
        # Set IEEE-compliant formatting
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='upper left', frameon=True)
        
        # Add value annotations
        for i, v in enumerate(mrr_scores):
            ax.text(i - bar_width/2, v + 0.03, f"{v:.3f}", 
                   ha='center', va='bottom', fontsize=8, rotation=0)
            
        for i, v in enumerate(ndcg_scores):
            ax.text(i + bar_width/2, v + 0.03, f"{v:.3f}", 
                   ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Add grid with specific styling
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limit for better visualization
        ax.set_ylim(0, 1.0)
        
        # Add figure number and reference text
        plt.figtext(0.01, 0.01, 'Figure 6: Performance comparison of different models.', 
                   ha='left', fontsize=9, fontstyle='italic')
        
        # Tight layout and save
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        # Save in multiple formats for academic publication
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(f"figures/model_comparison_{timestamp}.{fmt}", 
                       bbox_inches='tight', format=fmt)
        
        print(f"Generated model comparison figure: figures/model_comparison_{timestamp}.png")
    
    def generate_all_figures(self):
        """Generate all figures"""
        print("Generating all academic-grade figures...")
        
        # Load data once
        self.load_experiment_data()
        
        # Generate figures
        self.generate_competence_evolution_figure()
        self.generate_reward_evolution_figure()
        self.generate_comparative_performance_figure()
        
        print("All figures generated successfully in 'figures/' directory")

if __name__ == "__main__":
    # Create generator and generate figures
    generator = FinalFigureGenerator()
    generator.generate_all_figures()
