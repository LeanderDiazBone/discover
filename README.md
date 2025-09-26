# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning

This repository implements the methods and experiments presented in the "DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning" paper.
The implementation of the core off-policy RL algorithms and evaluation environments is adapted from the JaxGCRL repository (https://github.com/MichalBortkiewicz/JaxGCRL).

## Getting started

## Installation 📂

#### Editable Install (Recommended)

After cloning the repository, run one of the following commands.

With GPU on Linux:
```bash
pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> [!NOTE]  
> Make sure you have the correct CUDA version installed, i.e. CUDA >= 12.3.
> You can check your CUDA version with `nvcc --version` command.

With CPU on Mac:
```bash
export SDKROOT="$(xcrun --show-sdk-path)" # may be needed to build brax dependencies
pip install -e . 
```


### Documentation

Please refer to the JaxGCRL repository (https://github.com/MichalBortkiewicz/JaxGCRL) for the main documentation of the implementation and environments. The main addition we provide is a goal selection step in the beginning of each episodes, which guides the exploration. All the methods discussed in the paper are implemented and can be run as shown in `scripts/train.sh`.

### Reproducing the experiments

The `scripts` directory contains the `train.sh` script, which can be used to run the main experiments from the paper.

## Citation

```bibtex
@misc{diazbone2025discoverautomatedcurriculasparsereward,
      title={DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning}, 
      author={Leander Diaz-Bone and Marco Bagatella and Jonas Hübotter and Andreas Krause},
      year={2025},
      eprint={2505.19850},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.19850}, 
}
```
