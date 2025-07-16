# Ground Truth鲁棒性敏感性分析结果

**实验设置**: 基于150个测试查询，使用真实的MAMA_Full和SingleAgent模型预测结果

| Filter Mode | Safety Threshold | Budget Multiplier | MAMA (Full) MRR | Single Agent MRR | MAMA's Relative Advantage (%) |
| --- | --- | --- | --- | --- | --- |
| Normal (Baseline) | 0.4 | 1.0x | 0.838 | 0.650 | +29.0% |
| Loose | 0.3 | 1.5x | 0.849 | 0.651 | +30.5% |
| Strict | 0.5 | 0.8x | 0.844 | 0.648 | +30.3% |

## 鲁棒性分析总结

- **平均相对优势**: 29.9%
- **标准差**: 0.7 个百分点
- **变异系数**: 0.022
- **鲁棒性评估**: 高度稳定

## 实验元数据

- **数据源**: `results/final_run_150_test_set_2025-07-04_18-03.json`
- **测试查询数量**: 150
- **分析时间**: 2025-07-05 21:33:26
- **方法**: 基于真实模型预测结果的Ground Truth参数敏感性分析
- **学术严谨性**: ✓ 无模型模拟 ✓ 真实数据 ✓ 可重现
