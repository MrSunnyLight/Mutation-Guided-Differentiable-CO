# Differentiable Repairability & Mutations for Scalable Combinatorial Optimization (mQO)

[![arXiv](https://img.shields.io/badge/arXiv-2605.06921-b31b1b.svg)](https://arxiv.org/abs/2605.06921)

Paper link: [arXiv:2605.06921](https://arxiv.org/abs/2605.06921)

This repository contains Python code for solving **Maximum Independent Set (MIS)** and **Maximum Cut (MaxCut)** problems with continuous optimization.

The scripts optimize the variables of each graph directly. They do not train a neural network and do not need training data or pretrained weights.

## Files

The supplied code contains four scripts.

| File                               | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| [`mqo-mis.py`](mqo-mis.py)         | MIS with one candidate at a time     |
| [`pmqo-mis.py`](pmqo-mis.py)       | MIS with a batch of candidates.      |
| [`mqo-maxcut.py`](mqo-maxcut.py)   | MaxCut with one candidate at a time. |
| [`pmqo-maxcut.py`](pmqo-maxcut.py) | MaxCut with a batch of candidates.   |

## Installation

Use a Python environment supported by your PyTorch version. Python 3.10 or later is a reasonable starting point. 

### 1. Get the code

```bash
git clone https://github.com/MrSunnyLight/Mutation-Guided-Differentiable-CO.git
cd Mutation-Guided-Differentiable-CO
```

If you already downloaded the ZIP, extract it and open a terminal in the folder containing the four `.py` files.

### 2. Install packages

The code uses these packages: `torch`, `networkx`, `numpy`, `scipy`, `psutil`.

```bash
python -m pip install numpy scipy networkx psutil
```

Install PyTorch using the command from the [official installation page](https://pytorch.org/get-started/locally/). Select your operating system and the appropriate CPU or CUDA build. The CUDA choice should match your machine.

## Run

Run a script from the repository folder. For example, to solve MIS:

```bash
python mqo-mis.py
```

To solve MaxCut:

```bash
python mqo-maxcut.py
```

For the parallel versions, run `pmqo-mis.py` or `pmqo-maxcut.py` instead.

## Citation

If you use the method or code, cite the paper:

```bibtex
@article{sun2026differentiable,
  title   = {Differentiable Repairability \& Mutations for Scalable Combinatorial Optimization},
  author  = {Sun, Yongliang and Alkhouri, Ismail and Huang, Cheng-Han and
             Velasquez, Alvaro and Jha, Susmit and Wang, Rongrong},
  journal = {arXiv preprint arXiv:2605.06921},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.06921},
  url     = {https://arxiv.org/abs/2605.06921}
}
```
