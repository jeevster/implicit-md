# StABlE-Training
Repository to reproduce results in the paper [Stability-Aware Training of Machine Learning Force Fields with Differentiable Boltzmann Estimators](https://arxiv.org/abs/2402.13984), which introduces Stability-Aware Boltzmann Estimator (StABlE) Training for Machine Learning Force Fields (MLFFs). StABlE Training enables refinement of MLFFs using system observables (obtained via high-fidelity simulation or experiment) to improve their stability in long molecular dynamics (MD) simulations. We borrow some infrastructure code from the [MDsim](https://github.com/kyonofx/MDsim) codebase which accompanied the [Forces are Not Enough](https://arxiv.org/abs/2210.07237) paper (which is in turn based upon the [FAIR-Chem](https://github.com/FAIR-Chem/fairchem) repo from the Open Catalyst team at Meta). To a lesser extent, we also build upon the [TorchMD](https://github.com/torchmd/mdgrad) and [Neural Force Field](https://github.com/learningmatter-mit/NeuralForceField) codebases. 

## Environment Setup
TODO

## Download Data
- Run ```python download_stable_paper_data.py --download_path [DOWNLOAD_PATH]``` to download all required data for the systems considered in the paper (aspirin from [MD17](http://www.sgdml.org/#datasets), ac-Ala3-NHMe from [MD22](http://www.sgdml.org/#datasets), and Water from [Forces are Not Enough](https://arxiv.org/abs/2210.07237)). If you want to download other molecules (e.g benzene from MD17) or dataset sizes, you can run the corresponding system-specific file, e.g ```python data/md17.py --molecule benzene --size 50k```. Adding the ```--contiguous``` flag preserves time ordering (required to compute dynamical observables during StABlE Training).

## Quantum Mechanical Energy/Forces Pretraining

- ```pretraining.py```: Top-level script for traditional energy/forces QM training of NNIPs. This should be run to convergence prior to StABlE Training. 
- To train a SchNet model on aspirin 1k, run ```python pretraining.py --mode train --config-yml configs/pretraining/md17/schnet.yml --molecule aspirin --size 1k```. 
- To train a Nequip model on ac-Ala3-NHMe 1k, run ```python pretraining.py --mode train --config-yml configs/pretraining/md22/nequip.yml --molecule ac_Ala3_NHMe --size 1k --nequip```. 
- To train a GemNet-T model on Water 1k, run ```python pretraining.py --mode train --config-yml configs/pretraining/water/gemnet-T.yml --size 1k```. [TODO] we have provided checkpoints for all of these pretrained models in ```checkpoints/```.

## StABlE Training
```stable_training.py``` is the top-level script for running the StABlE Training algorithm.

- Training: run the script with the ```train.yml``` config. 
- To finetune a SchNet model pretrained on aspirin, run ```python stable_training.py --config-yml configs/stable_training/md17/train.yml```.
- To finetune the NequIP model pretrained on Ac-Ala3-NHMe, run ```python stable_training.py --config-yml configs/stable_training/md22/train.yml```. 
- For the GemNet-T model pretrained on water, run ```python stable_training.py --config-yml configs/stable_training/water/train.yml```. 
- [TODO] we have also provided the StABlE-trained checkpoints for each model/system in ```checkpoints/``` so that you can directly run the evaluation scripts (see next).

## Evaluation
- To evaluate the final performance of the StABlE-trained model in MD simulations, run the same scripts with the ```simulate.yml``` config. For example, ```python stable_training.py --config-yml configs/stable_training/md17/simulate.yml --eval_model [EVAL_MODEL]```.
  - The ```--eval_model``` flag specifies which checkpoint from StABlE Training to use for evaluation. Setting it to ```"pre"``` uses the QM-pretrained checkpoint (i.e before StABlE Training). Setting to a value of ```"10k"``` or       ```"50k"``` will load the QM-pretrained checkpoint on the 10k or 50k dataset respectively. Setting it to ```"post"``` uses the final checkpoint from StABlE Training. You can also set it to a value of ```"post_cycle[i]_epoch[j]"```, in which case the checkpoint from the jth epoch from the ith cycle of StABlE Training will be used. Setting to ```"post_cycle[i]"``` will load the checkpoint of the final epoch in cycle i. At the conclusion of evaluation, a file called ```final_metrics.json``` will be written to the ```log_dir``` specified in the config file, containing metrics relating to stability and observable accuracy of the simulation. [TODO] specify which eval_model to use for each setting. 

## Important Files
- ```simulator.py```: Defines the ```Simulator``` class which performs MD simulations with a neural network interatomic potential (NNIP).
- ```boltzmann_estimator.py```: Defines the ```BoltzmannEstimator``` class which computes the local and global N-sample Boltzmann Estimators (derived in the paper) necessary to train NNIPs to observables.
- ```mdsim/```: Contains main components of the MD engine, including NNIP model architectures, calculators, integrators, and observables. Largely built upon the [MDsim](https://github.com/kyonofx/MDsim) repo.
- ```configs/```: Contains configuration files for the three systems considered in the paper: Aspirin (MD17), ac-Ala3-NHMe (MD22), and Water. For StABlE Training, each system has a ```train.yml``` and ```simulate.yml``` for training and simulation evaluation respectively.
  
## Citation

```bibtex
@misc{raja2024stabilityawaretrainingmachinelearning,
      title={Stability-Aware Training of Machine Learning Force Fields with Differentiable Boltzmann Estimators}, 
      author={Sanjeev Raja and Ishan Amin and Fabian Pedregosa and Aditi S. Krishnapriyan},
      year={2024},
      eprint={2402.13984},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.13984}, 
}
