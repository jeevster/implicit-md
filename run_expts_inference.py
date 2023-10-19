import subprocess
from itertools import product

mol = 'aspirin'

config_yml = '/home/sanjeevr/implicit-md/configs/md17/md17.yml'
exp_name_1 = "TEST_original"
exp_name_2 = "TEST_alwayslearn"

# command = f"python nvt_implicit.py --config-yml {config_yml} \
#             --name=md17 --molecule={mol} --train=False --eval_model=post\
#             --exp_name={exp_name_1} --n_epochs=300 --bond_dev_tol=0.5"
# subprocess.run(command.split(" "))

command = f"python nvt_implicit.py --config-yml {config_yml} \
            --name=md17 --molecule={mol} --train=False --eval_model=post\
            --exp_name={exp_name_2} --n_epochs=300 --bond_dev_tol=0.5"
subprocess.run(command.split(" "))

command = f"python nvt_implicit.py --config-yml {config_yml} \
                    --name=md17 --molecule={mol} --train=False --eval_model=pre\
                    --exp_name={exp_name_1} --n_epochs=300 --bond_dev_tol=0.5"
subprocess.run(command.split(" "))





