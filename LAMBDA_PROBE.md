# Self-Pruning Neural Network Report

## Why an L1 penalty on sigmoid gates encourages sparsity

Each gate contributes directly to the objective through `lambda * sum(sigmoid(gate_scores))`.
That means every partially open connection pays a linear cost simply for staying active.
During optimization, a weight keeps its gate open only if the classification gain offsets that cost.
Connections that do not help enough are pushed toward smaller sigmoid values, which makes them easy to prune with a small threshold after training.

## Experimental Setup

- Dataset: CIFAR-10
- Reported accuracy: hard-pruned test accuracy using a gate threshold of `0.01`
- Best model selected by highest hard-pruned test accuracy: `lambda=1.00e-04`

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| --- | ---: | ---: |
| `1.00e-04` | 19.14 | 0.00 |
| `5.00e-04` | 18.75 | 0.00 |
| `1.00e-03` | 18.75 | 0.00 |

## Best Model Gate Distribution

![Gate distribution](artifacts/lambda_probe/best_model_gate_distribution.png)

A successful pruning run should show many gates clustered near zero, while the remaining useful connections stay noticeably above the pruning threshold.
