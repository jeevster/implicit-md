# StABlE-Training
PRIVATE/UNRELEASED repository for the paper [Stability-Aware Training of Neural Network Interatomic Potentials with Differentiable Boltzmann Estimators](https://arxiv.org/abs/2402.13984), which introduces Stability-Aware Boltzmann Estimator (StABlE) Training for Neural Network Interatomic Potentials (NNIPs). StABlE Training enables refinement of NNIPs using system observables (obtained via high-fidelity simulation or experiment) to improve their stability in long molecular dynamics (MD) simulations. We borrow some infrastructure code from the [MDsim](https://github.com/kyonofx/MDsim) codebase which accompanied the [Forces are Not Enough](https://arxiv.org/abs/2210.07237) paper, and to a lesser extent the [TorchMD](https://github.com/torchmd/mdgrad) and [Neural Force Field](https://github.com/learningmatter-mit/NeuralForceField) codebases. 

## Download Data
- Run ```python download_stable_paper_data.py --download_path [DOWNLOAD_PATH]``` to download all required data for the systems considered in the paper (aspirin from [MD17](http://www.sgdml.org/#datasets), ac-Ala3-NHMe from [MD22](http://www.sgdml.org/#datasets), and Water from [Forces are Not Enough](https://arxiv.org/abs/2210.07237)). If you want to download other molecules (e.g benzene from MD17) or dataset sizes, you can run the corresponding system-specific file, e.g ```python data/md17.py --molecule benzene --size 50k```. Adding the ```--contiguous``` flag preserves time ordering (required to compute dynamical observables during StABlE Training).

## Quantum Mechanical Energy/Forces Pretraining

- ```pretraining.py```: Top-level script for traditional energy/forces QM training of NNIPs. This should be run to convergence prior to StABlE Training. For example, to train a SchNet model on aspirin 1k, run ```python pretraining.py --mode train --config-yml configs/pretraining/md17/schnet.yml --molecule aspirin -size 1k```

## StABlE Training
```stable_training.py``` is the top-level script for running the StABlE Training algorithm.

- Training: run the script with the ```train.yml``` config. For example, to finetune a SchNet model pretrained on aspirin, run ```python stable_training.py --config-yml configs/stable_training/md17/train.yml```.
- Evaluation: to evaluate the final performance of the StABlE-trained model in MD simulations, run the script with the ```simulate.yml``` config. For example, ```python stable_training.py --config-yml configs/stable_training/md17/simulate.yml --eval_model [EVAL_MODEL]```.
  - The ```--eval_model``` flag specifies which checkpoint from StABlE Training to use for evaluation. Setting it to ```"pre"``` uses the QM-pretrained checkpoint (i.e before StABlE Training). Setting to a value of ```"10k"``` or       ```"50k"``` will load the QM-pretrained checkpoint on the 10k or 50k dataset respectively. Setting it to ```"post"``` uses the final checkpoint from StABlE Training. You can also set it to a value of ```"post_cycle[i]_epoch[j]"```, in which case the checkpoint from the jth epoch from the ith cycle of StABlE Training will be used. Setting to ```"post_cycle[i]"``` will load the checkpoint of the final epoch in cycle i. 
## Important Files
- ```simulator.py```: Defines the ```Simulator``` class which performs MD simulations with a neural network interatomic potential (NNIP).
- ```boltzmann_estimator.py```: Defines the ```BoltzmannEstimator``` class which computes the local and global N-sample Boltzmann Estimators (derived in the paper) necessary to train NNIPs to observables.
- ```mdsim/```: Contains main components of the MD engine, including NNIP model architectures, calculators, integrators, and observables. Largely built upon the [MDsim](https://github.com/kyonofx/MDsim) repo.
- ```configs/```: Contains configuration files for the three systems considered in the paper: Aspirin (MD17), ac-Ala3-NHMe (MD22), and Water. For StABlE Training, each system has a ```train.yml``` and ```simulate.yml``` for training and simulation evaluation respectively.

## Authors
- Sanjeev Raja
- Ishan Amin
- Fabian Pedregosa
- Aditi Krishnapriyan
  
## Abstract
Neural network interatomic potentials (NNIPs) are an attractive alternative to ab-initio methods for molecular dynamics (MD) simulations. However, they can produce unstable simulations which sample unphysical states, limiting their usefulness for modeling phenomena occurring over longer timescales. To address these challenges, we present Stability-Aware Boltzmann Estimator (StABlE) Training, a multi-modal training procedure which combines conventional supervised training from quantum-mechanical energies and forces with reference system observables, to produce stable and accurate NNIPs. StABlE Training iteratively runs MD simulations to seek out unstable regions, and corrects the instabilities via supervision with a reference observable. The training procedure is enabled by the Boltzmann Estimator, which allows efficient computation of gradients required to train neural networks to system observables, and can detect both global and local instabilities. We demonstrate our methodology across organic molecules, tetrapeptides, and condensed phase systems, along with using three modern NNIP architectures. In all three cases, StABlE-trained models achieve significant improvements in simulation stability and recovery of structural and dynamic observables. In some cases, StABlE-trained models outperform conventional models trained on datasets 50 times larger. As a general framework applicable across NNIP architectures and systems, StABlE Training is a powerful tool for training stable and accurate NNIPs, particularly in the absence of large reference datasets.

## Citation

```bibtex
@misc{raja2024stabilityaware,
  title={Stability-Aware Training of Neural Network Interatomic Potentials with Differentiable Boltzmann Estimators}, 
  author={Sanjeev Raja and Ishan Amin and Fabian Pedregosa and Aditi S. Krishnapriyan},
  year={2024},
  eprint={2402.13984},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
