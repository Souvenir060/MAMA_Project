# MAMA Framework Scalability Stress Test - Summary Table

**Test Date:** 2025-07-16 12:15:55

**Test Queries:** 150 queries

## Performance Summary

| N | Semantic Matching Latency (ms) | Registrar Throughput (msg/s) | Success Rate | MARL Decision Latency (ms) |
|---|---|---|---|---|
| 10 | 7.89±5.87 | 29713.7 | 100.0% | 0.047±0.033 |
| 50 | 16.32±5.16 | 92674.6 | 100.0% | 0.046±0.012 |
| 100 | 21.14±8.37 | 44176.7 | 100.0% | 0.042±0.005 |
| 500 | 18.32±5.55 | 71692.4 | 100.0% | 0.044±0.007 |
| 1000 | 19.78±9.50 | 60214.8 | 100.0% | 0.042±0.005 |
| 5000 | 23.59±9.41 | 62559.3 | 100.0% | 0.047±0.017 |

## Key Findings

- **Semantic Matching:** Latency ranges from 7.89ms (N=10) to 23.59ms (N=5000)
- **Registrar Service:** Throughput ranges from 29713.7 to 92674.6 msg/s
- **MARL Decision:** Latency ranges from 0.042ms to 0.047ms

## Technical Notes

- All latency measurements use high-precision timing (perf_counter)
- Semantic matching uses authentic SBERT model (all-MiniLM-L6-v2)
- Registrar service tested with 10,000 concurrent messages
- MARL decision tested with pre-trained neural network
- Results include statistical confidence intervals (mean ± std)
