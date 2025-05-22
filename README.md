# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning

This repository implements the methods and experiments presented in the "DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning" paper.
The implementation of the core off-policy RL algorithms and evaluation environments is adapted from the JaxGCRL repository (https://github.com/MichalBortkiewicz/JaxGCRL).

## Getting started

### Installation

Requires Python 3.10.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Documentation

Please refer to the JaxGCRL repository (https://github.com/MichalBortkiewicz/JaxGCRL) for the main documentation of the implementation and environments. The main addition we provide is a goal selection step in the beginning of each episodes, which guides the exploration. All the methods discussed in the paper are implemented and can be run as shown in scripts/train.sh.

### Reproducing the experiments

The `scripts` directory contains the train.sh script, which can be used to run the main experiments from the paper.