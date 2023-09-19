import subprocess
from itertools import product


molecules_md17 = ['aspirin', 'naphthalene', 'salicylic_acid']# 'ethanol', 'toluene', 'benzene', 'salicylic_acid', 'toluene', 'uracil']

molecules_md22 = ['DHA', 'ac_Ala3_NHMe', 'AT_AT', 'stachyose', 'AT_AT_CG_CG'] #'buckyball_catcher', 'double_walled_nanotube']

config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'

#MD17 no energy/force
for mol in molecules_md17:
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --name=md17 --size=1k --rdf_loss_weight=1 --vacf_loss_weight=0 --minibatch_size=80 \
                 --energy_loss_weight=0 --force_loss_weight=0 --exp_name=TEST_resumestrategy0.25_learn_continuous_no_energyforce"
    subprocess.run(command.split(" "))

#MD17 energy/force
# for mol in molecules_md17:
#     command = f"python nve_implicit.py --config-yml {config_yml} \
#                  --molecule={mol} --name=md17 --size=1k --rdf_loss_weight=1 --vacf_loss_weight=0 --minibatch_size=80 \
#                  --energy_loss_weight=0.05 --force_loss_weight=0.95 --exp_name=TEST_resumestrategy0.25_learn_continuous"
#     subprocess.run(command.split(" "))

# #MD17 no energy/force + dynamics
# for mol in molecules_md17:
#     command = f"python nve_implicit.py --config-yml {config_yml} \
#                  --molecule={mol} --results_dir=nnip_continuous_dynamics_new --name=md17 --size=1k \
#                  --integrator=Langevin --rdf_loss_weight=1 --vacf_loss_weight=1 --minibatch_size=80 \
#                  --energy_loss_weight=0.0 --force_loss_weight=0.0 --exp_name=TEST_resumestrategy0.25_learn_continuous_noenergyforce"
   
#     subprocess.run(command.split(" "))

# #MD22 no energy/force
# for mol in molecules_md22:
#     command = f"python nve_implicit.py --config-yml {config_yml} \
#                  --molecule={mol} --name=md22 --size=1k --rdf_loss_weight=1 \
#                  --vacf_loss_weight=0 --energy_loss_weight=0.0 --force_loss_weight=0.0 \
#                  --exp_name=TEST_resumestrategy0.25_learn_continuous_noenergyforce"
#     # if mol in ['double_walled_nanotube', 'buckyball_catcher']:
#     #     command += " --minibatch_size=20"
                
#     subprocess.run(command.split(" "))


