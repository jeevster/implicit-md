import subprocess
from itertools import product


molecules_md17 = ['aspirin', 'naphthalene'] #ethanol', 'toluene', 'benzene', 'salicylic_acid', 'toluene', 'uracil']

molecules_md22 = ['DHA', 'ac_Ala3_NHMe', 'AT_AT', 'stachyose', 'AT_AT_CG_CG'] #'buckyball_catcher', 'double_walled_nanotube']

config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for mol in molecules_md17:
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --name=md17 --size=1k --rdf_loss_weight=1 --vacf_loss_weight=0 --minibatch_size=80"
    subprocess.run(command.split(" "))

# for mol in molecules_md17:
#     command = f"python nve_implicit.py --config-yml {config_yml} \
#                  --molecule={mol} --results_dir=nnip_continuous_dynamics_new --name=md17 --size=1k --integrator=Langevin --rdf_loss_weight=1 --vacf_loss_weight=1 --minibatch_size=80"
   
#     subprocess.run(command.split(" "))

for mol in molecules_md22:
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --name=md22 --size=1k --rdf_loss_weight=1 --vacf_loss_weight=0"
    if mol in ['double_walled_nanotube', 'buckyball_catcher']:
        command += " --minibatch_size=20"
                
    subprocess.run(command.split(" "))


