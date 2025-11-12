# CircScale
This repository is the corresponding model repository for the paper "CircScale: Learning Scale-Adaptive Circuit Representations for Logic Synthesis Optimization"

## Abstract
Logic Synthesis Optimization systematically transforms And-Inverter Graph (AIG) circuits through representation-guided transformations to improve circuit quality, serving as a critical step in Electronic Design Automation (EDA) workflows. However, real-world logic synthesis involves circuits with highly heterogeneous scales and dynamically evolving topologies. These characteristics pose significant challenges for previous circuit representation learning methods, which often struggle with the heterogeneous scale bottleneck and fail to capture the dynamic nature of optimization in static graph representations. This paper proposes CircScale, a framework that learns scale-adaptive circuit representations for logic synthesis optimization. Specifically, CircScale employs multi-hop neighborhood aggregation with circuit-aware gating to dynamically balance structural features across heterogeneous scales. It further introduces an early graph-sequence interaction mechanism that models circuit evolution and dynamically guides the optimization process. Comprehensive experiments are performed on 124,500 circuit-sequence pairs collected from five benchmarks, and the results demonstrate the superior representation capability of CircScale, achieving 14.89\%-38.39\% delay reduction and 23.89\%-38.87\% area improvement over state-of-the-art methods. These results confirm that CircScale effectively learns comprehensive and adaptive circuit representations for logic synthesis optimization.

## Requirements
Python >= 3.8

PyTorch >= 2.1.2

PyTorch Geometric >= 2.6.1

## Sequence Generation Details:
Number of sequences per circuit: 1,500

Sequence length: 20 steps

Total samples: 124,500 circuit-sequence pairs (83 circuits × 1,500 sequences)

QoR metrics: Delay and area recorded after each optimization step

## Dataset Selection
To comprehensively evaluate CircScale's prediction performance and generalization capability on heterogeneous-scale circuits, this study employs five public benchmark circuit datasets covering circuit designs of different scales, complexity, and application domains. The datasets include: EPFL, ISCAS85, ISCAS89, OpenABC-D, and Opencores, covering a wide range of circuit types from small-scale combinational circuits to large-scale industrial designs, exhibiting significant heterogeneity in circuit scale, topological structure complexity, and application scenarios.

To ensure the representativeness and reproducibility of experiments, this study conducted systematic screening of circuits in each benchmark collection. Selection criteria include: (1) representativeness of circuit scale, ensuring coverage of small, medium, and large scale ranges; (2) diversity of topological structure, including different depth-to-width ratios and connection densities; (3) typicality of functional characteristics, covering different logic function types. After screening, 83 representative circuit designs were retained as evaluation benchmarks.

EPFL selects 11 representative circuits, including arithmetic circuits such as adder, multiplier, sin, square, log2, and max, as well as control circuits such as arbiter, mem\_ctrl, voter, bar, and cavlc.This dataset is mainly applied in arithmetic operations, communication protocols, and control logic domains, with node scales ranging from hundreds to tens of thousands. The notable characteristics of this dataset are its extremely high topological depth and scale heterogeneity, providing a rigorous test benchmark for evaluating model performance on deep-level structures and cross-scale circuits.

ISCAS85 contains 10 combinational logic benchmark circuits, including c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, and c7552. Since its release in 1985, this dataset has been a standard benchmark in automatic test pattern generation (ATPG) and logic synthesis domains. This dataset belongs to small-scale combinational logic circuit collections, and its compact topological structure and relatively shallow logic depth are suitable for evaluating the model's capability to capture global structural features in small-scale circuits.

ISCAS89 selects 8 representative sequential circuits: s1238, s1423, s1488, s5378, s9234, s15850, s35932, and s38417. Compared to ISCAS85, it introduces the complexity of sequential logic, while the circuit scale span is larger, widely applied in sequential test generation, scan testing, and partial scan technology research.

OpenABC-D, as a large-scale machine learning dataset, selects 36 representative industrial-level circuits, covering multiple categories including communication/bus protocols (ac97\_ctrl, ethernet, pci, wb\_conmax, wb\_dma), encryption algorithms (aes series, des3\_area, sha256), digital signal processing (fir, iir), and processors (bp\_be, picosoc, tinyRocket, tv80). This dataset represents the complexity and diversity of real industrial designs, being not only the largest-scale dataset but also the dataset with the strongest scale heterogeneity, providing the most rigorous test environment for evaluating model performance in large-scale practical applications and cross-scale generalization capability.

Opencores comes from the OpenCores open-source community, which has been the main online platform for developing gate-level IP cores since 1999. We select 18 open-source hardware IP cores, including communication interfaces (ac97\_ctrl, i2c, simple\_spi, spi, usb\_phy), encryption algorithms (aes\_core, des\_area, systemcaes, systemcdes), bus protocols (pci\_bridge32, pci\_conf\_cyc\_addr\_dec, pci\_spoci\_ctrl, wb\_conmax, wb\_dma), controllers (mem\_ctrl, sasc, steppermotordrive), and processors (tv80), providing rich test scenarios for evaluating model performance on open-source hardware designs.

## Data Split
Recipe-Inductive Setup:

Training set: 66% (randomly selected recipes)

Validation set: 34% (non-overlapping recipes)

All circuits are visible in both training and testing phases, Optimization sequences are completely non-overlapping across splits

## Model Training
### Delay Prediction
```python
python train.py --dataset epfl --metric Delay --use_dynamic_interaction --use_multiscale
```
### Area Prediction
```python
python train.py --dataset epfl --metric Area --use_dynamic_interaction --use_multiscale
```

## Model Inference
```python
python inference.py --model_path --vocab_path --metric Delay --dataset epfl --use_multiscale --use_dynamic_interaction
python inference.py --model_path --vocab_path --metric Area --dataset epfl --use_multiscale --use_dynamic_interaction
```
