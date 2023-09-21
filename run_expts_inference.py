import subprocess
from itertools import product


molecules1 = ['naphthalene']
molecules2 = ['salicylic_acid']
molecules3 = ['aspirin']
# molecules1 = ['DHA']
# molecules2 = ['ac_Ala3_NHMe']

config_yml = '/global/u2/s/sanjeevr/implicit-md/configs/md17/base.yml'
exp_name_1 = "TEST_resumestrategy0.25_learn_continuous_no_energyforce"
exp_name_2 = "TEST_resumestrategy0.25_learn_continuous"
for exp_name in [exp_name_1, exp_name_2]:
    for mol in molecules3:
        command = f"python nve_implicit.py --config-yml {config_yml} \
                    --name=md17 --molecule={mol} --train=False --eval_model=post\
                    --exp_name={exp_name} --n_epochs=300 --bond_dev_tol=0.5"
        subprocess.run(command.split(" "))

        command = f"python nve_implicit.py --config-yml {config_yml} \
                    --name=md17 --molecule={mol} --train=False --eval_model=pre\
                    --exp_name={exp_name} --n_epochs=300 --bond_dev_tol=0.5"
        subprocess.run(command.split(" "))



