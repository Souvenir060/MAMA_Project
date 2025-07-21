# Ground Truth Robustness Analysis Results
Generated: 2025-07-18 15:48:35

## Filter Mode Configurations
| Mode | Safety Threshold | Budget Multiplier | Description |
|------|------------------|-------------------|-------------|
| Normal | 0.4 | 1.0x | Baseline mode - paper default parameters |
| Loose | 0.3 | 1.5x | Loose mode - more candidate flights enter ranking stage |
| Strict | 0.5 | 0.8x | Strict mode - fewer candidate flights, simpler ranking problem |

## Performance Results by Mode
| Filter Mode | Model | MRR | NDCG@5 | ART (s) |
|-------------|-------|-----|--------|---------|
| Normal | MAMA (Full) | 0.843 | 0.848 | 2.151 |
| Normal | MAMA (No Trust) | 0.748 | 0.781 | 1.847 |
| Normal | Single Agent | 0.646 | 0.682 | 1.508 |
| Normal | Traditional Ranking | 0.556 | 0.581 | 1.159 |
| Loose | MAMA (Full) | 0.848 | 0.857 | 2.175 |
| Loose | MAMA (No Trust) | 0.741 | 0.775 | 1.820 |
| Loose | Single Agent | 0.651 | 0.674 | 1.510 |
| Loose | Traditional Ranking | 0.553 | 0.586 | 1.190 |
| Strict | MAMA (Full) | 0.840 | 0.844 | 2.151 |
| Strict | MAMA (No Trust) | 0.735 | 0.775 | 1.853 |
| Strict | Single Agent | 0.645 | 0.672 | 1.499 |
| Strict | Traditional Ranking | 0.555 | 0.581 | 1.150 |

## MAMA (Full) vs Single Agent Advantage Analysis
| Filter Mode | MRR Advantage (%) | NDCG@5 Advantage (%) | ART Advantage (%) |
|-------------|-------------------|----------------------|-------------------|
| Normal | 30.4% | 24.4% | -29.9% |
| Loose | 30.3% | 27.2% | -30.6% |
| Strict | 30.3% | 25.7% | -30.3% |

## Robustness Metrics
| Metric | Mean Advantage | Std Dev | Coefficient of Variation | Robustness Level |
|--------|----------------|---------|--------------------------|------------------|
| MRR_robustness | 30.3% | 0.1pp | 0.002 | Very High |
| NDCG_robustness | 25.8% | 1.2pp | 0.045 | Very High |
| ART_robustness | -30.3% | 0.3pp | -0.009 | Very High |
