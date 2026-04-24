# Self-Pruning Neural Network Report

## Why an L1 penalty on sigmoid gates encourages sparsity

Each gate contributes directly to the objective through `lambda * sum(sigmoid(gate_scores))`.
That means every partially open connection pays a linear cost simply for staying active.
During optimization, a weight keeps its gate open only if the classification gain offsets that cost.
Connections that do not help enough are pushed toward smaller sigmoid values, which makes them easy to prune with a small threshold after training.

## Experimental Setup

- Dataset: FakeData
- Reported accuracy: hard-pruned test accuracy using a gate threshold of `0.01`
- Best model selected by highest hard-pruned test accuracy: `lambda=0.00e+00`

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| --- | ---: | ---: |
| `0.00e+00` | 12.50 | 0.00 |
| `1.00e-06` | 12.50 | 0.00 |
| `5.00e-06` | 12.50 | 0.00 |

## Best Model Gate Distribution

![Gate distribution](artifacts/smoke/best_model_gate_distribution.png)

A successful pruning run should show many gates clustered near zero, while the remaining useful connections stay noticeably above the pruning threshold.
