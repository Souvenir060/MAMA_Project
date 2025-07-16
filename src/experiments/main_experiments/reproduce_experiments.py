#!/usr/bin/env python3
"""
运行复现
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """打印美观的标题"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}")
    print(f"📝 执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 成功完成: {description}")
            if result.stdout:
                print(f"📊 输出:\n{result.stdout}")
        else:
            print(f"❌ 执行失败: {description}")
            print(f"错误信息: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False

def check_dependencies():
    """检查依赖环境"""
    print_header("步骤1: 环境依赖检查")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包 (修复导入名称)
    required_packages = [
        ('numpy', 'numpy'), 
        ('scipy', 'scipy'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'), 
        ('sklearn', 'scikit-learn'),  # 修复：导入名是sklearn，显示名是scikit-learn
        ('transformers', 'transformers')
    ]
    
    missing_packages = []
    for import_name, display_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name} - 已安装")
        except ImportError:
            print(f"❌ {display_name} - 缺失")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  缺失包: {', '.join(missing_packages)}")
        print("请先安装: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖包检查完成")
    return True

def verify_project_structure():
    """验证项目结构"""
    print_header("步骤2: 项目结构验证")
    
    required_dirs = ['models', 'evaluation', 'data', 'results']
    required_files = [
        'test_experiments.py',
        'hyperparameter_optimization.py', 
        'create_academic_figures.py',
        'models/base_model.py',
        'models/mama_full.py',
        'models/traditional_ranking.py',
        'models/single_agent_system.py'
    ]
    
    all_exists = True
    
    # 检查目录
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ 目录存在: {dir_name}/")
        else:
            print(f"❌ 目录缺失: {dir_name}/")
            all_exists = False
    
    # 检查文件
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"❌ 文件缺失: {file_name}")
            all_exists = False
    
    return all_exists

def run_core_experiments():
    """运行核心实验"""
    print_header("步骤3: 核心实验执行")
    
    print("🚀 开始运行核心学术实验...")
    print("⏱️  预计耗时: 3-5分钟")
    
    success = run_command(
        "python test_experiments.py",
        "执行主要的基线模型对比实验"
    )
    
    if success:
        print("\n📋 实验结果摘要:")
        if Path('results/academic_experiment_validation.json').exists():
            print("✅ 结果文件已生成: results/academic_experiment_validation.json")
        else:
            print("⚠️  结果文件未找到")
    
    return success

def run_hyperparameter_analysis():
    """运行超参数分析"""
    print_header("步骤4: 超参数敏感性分析")
    
    success = run_command(
        "python hyperparameter_optimization.py",
        "执行超参数网格搜索和敏感性分析"
    )
    
    return success

def generate_figures():
    """生成学术图表"""
    print_header("步骤5: 生成论文级图表")
    
    success = run_command(
        "python create_academic_figures.py",
        "生成性能对比、参数敏感性和统计显著性图表"
    )
    
    if success:
        figures_dir = Path('figures')
        if figures_dir.exists():
            figures = list(figures_dir.glob('*.png'))
            print(f"\n📊 生成的图表文件 ({len(figures)}个):")
            for fig in figures:
                print(f"   • {fig.name}")
        else:
            print("⚠️  figures目录未找到")
    
    return success

def display_results():
    """显示最终结果"""
    print_header("步骤6: 实验结果展示")
    
    # 读取并显示关键结果
    results_file = Path('results/academic_experiment_validation.json')
    if results_file.exists():
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("🏆 关键学术发现:")
        for finding in data['academic_conclusions']['key_findings']:
            print(f"   • {finding}")
        
        print("\n📊 性能统计:")
        for stat in data['performance_statistics']:
            print(f"   {stat['model']}: MRR={stat['MRR_mean']:.4f}±{stat['MRR_std']:.3f}")
        
        print("\n📈 统计显著性:")
        for test in data['significance_tests'][:3]:  # 显示前3个重要对比
            status = "✅ 显著" if test['significant'] else "❌ 不显著"
            print(f"   {test['comparison']}: p={test['p_value']:.2e} {status}")
        
        print("\n🔍 最佳超参数配置:")
        best_config = data['hyperparameter_sensitivity']['best_configuration']
        print(f"   α={best_config['alpha']}, β={best_config['beta']}, MRR={best_config['MRR']:.4f}")
    else:
        print("❌ 实验结果文件未找到")

def main():
    """主函数 - 完整实验复现流程"""
    print("🚀 MAMA系统学术实验完整复现")
    print("🎓 为顶级学术会议设计的严谨实验验证")
    print(f"📅 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 检查环境
    if not check_dependencies():
        print("❌ 环境检查失败，请先安装缺失的依赖包")
        return False
    
    # 步骤2: 验证项目结构
    if not verify_project_structure():
        print("❌ 项目结构不完整，请检查缺失的文件")
        return False
    
    # 步骤3: 运行核心实验
    if not run_core_experiments():
        print("❌ 核心实验执行失败")
        return False
    
    # 步骤4: 超参数分析（可选）
    print("\n❓ 是否运行完整的超参数分析？(输入 y/n, 默认跳过)")
    try:
        user_input = input().strip().lower()
        if user_input == 'y':
            run_hyperparameter_analysis()
        else:
            print("⏭️  跳过超参数分析（使用已有结果）")
    except (EOFError, KeyboardInterrupt):
        print("⏭️  跳过超参数分析（使用已有结果）")
    
    # 步骤5: 生成图表
    if not generate_figures():
        print("❌ 图表生成失败")
        return False
    
    # 步骤6: 显示结果
    display_results()
    
    print_header("🎉 实验复现完成！")
    print("📁 输出文件:")
    print("   • results/academic_experiment_validation.json (实验数据)")
    print("   • figures/*.png (论文图表)")
    print("   • figures/*.pdf (高质量PDF)")
    
    print("\n💡 下一步:")
    print("   1. 查看 figures/ 目录中的论文级图表")
    print("   2. 分析 results/ 目录中的详细数据")
    print("   3. 参考实验结果撰写学术论文")
    
    print(f"\n✅ 实验验证成功完成! (耗时: {time.strftime('%H:%M:%S')})")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 