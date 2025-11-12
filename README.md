# CircScale
This repository is the corresponding model repository for the paper "CircScale: Learning Scale-Adaptive Circuit Representations for Logic Synthesis Optimization"
## Abstract
Logic Synthesis Optimization systematically transforms And-Inverter Graph (AIG) circuits through representation-guided transformations to improve circuit quality, serving as a critical step in Electronic Design Automation (EDA) workflows. However, real-world logic synthesis involves circuits with highly heterogeneous scales and dynamically evolving topologies. These characteristics pose significant challenges for previous circuit representation learning methods, which often struggle with the heterogeneous scale bottleneck and fail to capture the dynamic nature of optimization in static graph representations. This paper proposes CircScale, a framework that learns scale-adaptive circuit representations for logic synthesis optimization. Specifically, CircScale employs multi-hop neighborhood aggregation with circuit-aware gating to dynamically balance structural features across heterogeneous scales. It further introduces an early graph-sequence interaction mechanism that models circuit evolution and dynamically guides the optimization process. Comprehensive experiments are performed on 124,500 circuit-sequence pairs collected from five benchmarks, and the results demonstrate the superior representation capability of CircScale, achieving 14.89\%-38.39\% delay reduction and 23.89\%-38.87\% area improvement over state-of-the-art methods. These results confirm that CircScale effectively learns comprehensive and adaptive circuit representations for logic synthesis optimization.
## Requirements
Python >= 3.8
PyTorch >= 2.1.2
PyTorch Geometric >= 2.6.1
## Project Structure
CircScale/
├── config.py                # Configuration files
├── data
├── datasets/
│   ├── EPFL
│   ├── ISCAS85_bench
│   ├── ISCAS89_bench
│   ├── OpenABC-D
│   └── OpenCores_bench
├── data_files
├── data_files_epfl
├── data_files_iscas85
├── data_files_iscas89
├── data_files_openabc
├── data_files_opencores
├── models/                  # Model implementations
│   └── circscale
├── utils/                   # Helper functions
│   ├── _init_.py
│   ├── helper.py
│   ├── visualization.py
│   └── metrics
├── train.py                 # Training script
└── inference.py             # Inference script
# Sequence Generation Details:
Number of sequences per circuit: 1,500
Sequence length: 20 steps
Total samples: 124,500 circuit-sequence pairs (83 circuits × 1,500 sequences)
QoR metrics: Delay and area recorded after each optimization step
#Data Split
Recipe-Inductive Setup:
Training set: 66% (randomly selected recipes)
Validation set: 34% (non-overlapping recipes)
All circuits are visible in both training and testing phases, Optimization sequences are completely non-overlapping across splits
