# MAMA Framework Scalability Stress Test - Summary Table

**Test Date:** 2025-07-16 15:39:05

**Test Queries:** 150 queries

## Performance Summary

| N | Semantic Matching Latency (ms) | Registrar Throughput (msg/s) | Success Rate | MARL Decision Latency (ms) |
|---|---|---|---|---|
| 10 | 10.28±26.30 | 32883.2 | 100.0% | 0.048±0.046 |
| 50 | 13.38±3.56 | 39052.4 | 100.0% | 0.042±0.008 |
| 100 | 14.55±3.07 | 48590.7 | 100.0% | 0.043±0.008 |
| 500 | 15.21±2.65 | 73756.6 | 100.0% | 0.043±0.008 |
| 1000 | 15.42±3.24 | 73257.0 | 100.0% | 0.043±0.006 |
| 5000 | 22.55±13.06 | 45684.8 | 100.0% | 0.043±0.008 |

## Key Findings

- **Semantic Matching:** Latency ranges from 10.28ms (N=10) to 22.55ms (N=5000)
- **Registrar Service:** Throughput ranges from 32883.2 to 73756.6 msg/s
- **MARL Decision:** Latency ranges from 0.042ms to 0.048ms

## Technical Notes

- All latency measurements use high-precision timing (perf_counter)
- Semantic matching uses authentic SBERT model (all-MiniLM-L6-v2)
- Registrar service tested with 10,000 concurrent messages
- MARL decision tested with pre-trained neural network
- Results include statistical confidence intervals (mean ± std)
