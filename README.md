# CircScale
This repository is the corresponding model repository for the paper "CircScale: Learning Scale-Adaptive Circuit Representations for Logic Synthesis Optimization"

## Abstract
Logic Synthesis Optimization systematically transforms And-Inverter Graph (AIG) circuits through representation-guided transformations to improve circuit quality, serving as a critical step in Electronic Design Automation (EDA) workflows. However, real-world logic synthesis involves circuits with highly heterogeneous scales and dynamically evolving topologies. These characteristics pose significant challenges for previous circuit representation learning methods, which often struggle with the heterogeneous scale bottleneck and fail to capture the dynamic nature of optimization in static graph representations. This paper proposes CircScale, a framework that learns scale-adaptive circuit representations for logic synthesis optimization. Specifically, CircScale employs multi-hop neighborhood aggregation with circuit-aware gating to dynamically balance structural features across heterogeneous scales. It further introduces an early graph-sequence interaction mechanism that models circuit evolution and dynamically guides the optimization process. Comprehensive experiments are performed on 124,500 circuit-sequence pairs collected from five benchmarks, and the results demonstrate the superior representation capability of CircScale, achieving 14.89\%-38.39\% delay reduction and 23.89\%-38.87\% area improvement over state-of-the-art methods. These results confirm that CircScale effectively learns comprehensive and adaptive circuit representations for logic synthesis optimization.

## Requirements
Python >= 3.8

PyTorch >= 2.1.2

PyTorch Geometric >= 2.6.1

# Sequence Generation Details:
Number of sequences per circuit: 1,500

Sequence length: 20 steps

Total samples: 124,500 circuit-sequence pairs (83 circuits × 1,500 sequences)

QoR metrics: Delay and area recorded after each optimization step

# Data Split
Recipe-Inductive Setup:

Training set: 66% (randomly selected recipes)

Validation set: 34% (non-overlapping recipes)

All circuits are visible in both training and testing phases, Optimization sequences are completely non-overlapping across splits

# Model Training
## Delay Prediction
```python
python train.py --dataset epfl --metric Delay --use_dynamic_interaction --use_multiscale
```
## Area Prediction
```python
python train.py --dataset epfl --metric Area --use_dynamic_interaction --use_multiscale
```

# Model Inference
```python
python inference.py --model_path --vocab_path --metric Delay --dataset epfl --use_multiscale --use_dynamic_interaction
python inference.py --model_path --vocab_path --metric Area --dataset epfl --use_multiscale --use_dynamic_interaction
```
