#!/usr/bin/env python3
"""
Script for drawing the growth curve of intelligent agent capability
=========================

Based on the capability evolution log file, draw the learning and growth curves of each intelligent agent.
Demonstrate the effectiveness of the trust mechanism in the MAMA framework and the learning ability of the intelligent agent.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_competence_data(log_file):
    """加载能力演进数据"""
    try:
        df = pd.read_csv(log_file)
        print(f"✅ 成功加载数据文件: {log_file}")
        print(f"📊 数据概况: {len(df)} 条记录，{df['Interaction_ID'].nunique()} 次交互")
        return df
    except Exception as e:
        print(f"❌ 加载数据文件失败: {e}")
        return None

def plot_competence_evolution(df, output_dir="figures", show_plot=True):
    """绘制智能体能力演进曲线"""
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 设置绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAMA框架智能体能力演进分析\nMulti-Agent Model AI Framework - Agent Competence Evolution', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 颜色配置
    colors = {
        'safety_agent_competence': '#FF6B6B',
        'economic_agent_competence': '#4ECDC4', 
        'weather_agent_competence': '#45B7D1',
        'flight_info_agent_competence': '#96CEB4',
        'integration_agent_competence': '#FFEAA7'
    }
    
    # 中文智能体名称
    agent_names = {
        'safety_agent_competence': '安全评估智能体',
        'economic_agent_competence': '经济分析智能体',
        'weather_agent_competence': '天气分析智能体',
        'flight_info_agent_competence': '航班信息智能体',
        'integration_agent_competence': '集成协调智能体'
    }
    
    # 子图1: 所有智能体能力演进
    ax1 = axes[0, 0]
    for agent_name in colors.keys():
        agent_data = df[df['Agent_Name'] == agent_name]
        if not agent_data.empty:
            ax1.plot(agent_data['Interaction_ID'], agent_data['Competence_Score'], 
                    color=colors[agent_name], linewidth=2.5, 
                    label=agent_names[agent_name], alpha=0.8)
            
            # 添加趋势线
            z = np.polyfit(agent_data['Interaction_ID'], agent_data['Competence_Score'], 2)
            p = np.poly1d(z)
            ax1.plot(agent_data['Interaction_ID'], p(agent_data['Interaction_ID']), 
                    color=colors[agent_name], linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('交互次数')
    ax1.set_ylabel('能力分数')
    ax1.set_title('整体能力演进曲线')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.0)
    
    # 子图2: 核心智能体对比（安全 vs 经济）
    ax2 = axes[0, 1]
    key_agents = ['safety_agent_competence', 'economic_agent_competence']
    for agent_name in key_agents:
        agent_data = df[df['Agent_Name'] == agent_name]
        if not agent_data.empty:
            ax2.plot(agent_data['Interaction_ID'], agent_data['Competence_Score'], 
                    color=colors[agent_name], linewidth=3, 
                    label=agent_names[agent_name], marker='o', markersize=3, alpha=0.8)
            
            # 添加滑动平均线
            if len(agent_data) > 10:
                rolling_mean = agent_data['Competence_Score'].rolling(window=10, center=True).mean()
                ax2.plot(agent_data['Interaction_ID'], rolling_mean, 
                        color=colors[agent_name], linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('交互次数')
    ax2.set_ylabel('能力分数')
    ax2.set_title('核心智能体能力对比')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)
    
    # 子图3: 能力提升幅度分析
    ax3 = axes[1, 0]
    improvements = []
    agent_labels = []
    
    for agent_name in colors.keys():
        agent_data = df[df['Agent_Name'] == agent_name]
        if not agent_data.empty and len(agent_data) > 1:
            initial_score = agent_data['Competence_Score'].iloc[0]
            final_score = agent_data['Competence_Score'].iloc[-1]
            improvement = final_score - initial_score
            improvements.append(improvement)
            agent_labels.append(agent_names[agent_name])
    
    bars = ax3.bar(range(len(improvements)), improvements, 
                   color=[colors[list(colors.keys())[i]] for i in range(len(improvements))],
                   alpha=0.7)
    ax3.set_xlabel('智能体')
    ax3.set_ylabel('能力提升幅度')
    ax3.set_title('智能体学习效果对比')
    ax3.set_xticks(range(len(agent_labels)))
    ax3.set_xticklabels(agent_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{improvement:+.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 系统整体能力趋势
    ax4 = axes[1, 1]
    
    # 计算每次交互的系统平均能力
    system_competence = df.groupby('Interaction_ID')['Competence_Score'].mean().reset_index()
    
    ax4.plot(system_competence['Interaction_ID'], system_competence['Competence_Score'], 
            color='#2E86C1', linewidth=4, label='系统平均能力', alpha=0.8)
    
    # 添加趋势线
    z = np.polyfit(system_competence['Interaction_ID'], system_competence['Competence_Score'], 2)
    p = np.poly1d(z)
    ax4.plot(system_competence['Interaction_ID'], p(system_competence['Interaction_ID']), 
            color='red', linestyle='--', linewidth=2, label='趋势线', alpha=0.7)
    
    # 添加信心区间
    rolling_std = df.groupby('Interaction_ID')['Competence_Score'].std().reset_index()
    ax4.fill_between(system_competence['Interaction_ID'], 
                    system_competence['Competence_Score'] - rolling_std['Competence_Score'],
                    system_competence['Competence_Score'] + rolling_std['Competence_Score'],
                    alpha=0.2, color='#2E86C1')
    
    ax4.set_xlabel('交互次数')
    ax4.set_ylabel('平均能力分数')
    ax4.set_title('系统整体能力演进')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.0, 1.0)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = Path(output_dir) / 'agent_competence_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 能力演进图表已保存: {output_file}")
    
    # 同时保存PDF版本
    pdf_file = Path(output_dir) / 'agent_competence_evolution.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📄 PDF版本已保存: {pdf_file}")
    
    if show_plot:
        plt.show()
    
    return output_file

def generate_statistics_report(df, output_dir="results"):
    """生成详细的统计报告"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    agent_names = {
        'safety_agent_competence': '安全评估智能体',
        'economic_agent_competence': '经济分析智能体',
        'weather_agent_competence': '天气分析智能体',
        'flight_info_agent_competence': '航班信息智能体',
        'integration_agent_competence': '集成协调智能体'
    }
    
    report = f"""
MAMA框架智能体能力演进统计报告
============================================================
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据来源: 智能体交互日志
总交互次数: {df['Interaction_ID'].nunique()}
总记录数: {len(df)}

智能体性能分析:
============================================================
"""
    
    for agent_name, display_name in agent_names.items():
        agent_data = df[df['Agent_Name'] == agent_name]['Competence_Score']
        
        if not agent_data.empty:
            initial_score = agent_data.iloc[0]
            final_score = agent_data.iloc[-1]
            improvement = final_score - initial_score
            mean_score = agent_data.mean()
            std_score = agent_data.std()
            min_score = agent_data.min()
            max_score = agent_data.max()
            
            # 计算学习趋势
            x = np.arange(len(agent_data))
            if len(agent_data) > 1:
                slope, _ = np.polyfit(x, agent_data, 1)
                trend = "上升" if slope > 0.001 else "下降" if slope < -0.001 else "稳定"
            else:
                slope = 0
                trend = "无数据"
            
            report += f"""
{display_name}:
  初始能力: {initial_score:.4f}
  最终能力: {final_score:.4f}
  改进幅度: {improvement:+.4f}
  平均能力: {mean_score:.4f}
  标准差: {std_score:.4f}
  最低能力: {min_score:.4f}
  最高能力: {max_score:.4f}
  学习趋势: {trend} (斜率: {slope:.6f})
"""
    
    # 系统级别分析
    if df['Interaction_ID'].nunique() > 1:
        first_interaction = df['Interaction_ID'].min()
        last_interaction = df['Interaction_ID'].max()
        
        system_initial = df[df['Interaction_ID'] == first_interaction]['Competence_Score'].mean()
        system_final = df[df['Interaction_ID'] == last_interaction]['Competence_Score'].mean()
        system_improvement = system_final - system_initial
        
        report += f"""
系统级别分析:
============================================================
系统整体能力:
  初始平均: {system_initial:.4f}
  最终平均: {system_final:.4f}
  系统改进: {system_improvement:+.4f}
  
学习效果评估:
  - 交互次数: {df['Interaction_ID'].nunique()}
  - 数据完整性: {len(df) / (df['Interaction_ID'].nunique() * 5) * 100:.1f}%
  - 系统整体趋势: {'上升' if system_improvement > 0.01 else '下降' if system_improvement < -0.01 else '稳定'}
============================================================
"""
    
    # 保存报告
    report_file = Path(output_dir) / 'competence_evolution_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 统计报告已保存: {report_file}")
    print(report)
    
    return report_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='绘制智能体能力演进曲线')
    parser.add_argument('--log_file', '-f', default='competence_evolution_realistic.log',
                       help='能力演进日志文件路径')
    parser.add_argument('--output_dir', '-o', default='figures',
                       help='输出目录')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图表，仅保存文件')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_competence_data(args.log_file)
    if df is None:
        return
    
    # 绘制图表
    plot_competence_evolution(df, args.output_dir, not args.no_show)
    
    # 生成统计报告
    generate_statistics_report(df)
    
    print("\n🎉 智能体能力演进分析完成！")
    print("📊 图表和报告已生成，展示了MAMA框架的学习效果")

if __name__ == "__main__":
    main() 