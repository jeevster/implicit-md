import subprocess
from itertools import product


molecules_md17 = ['aspirin', 'naphthalene', 'salicylic_acid']
molecules2 = ['aspirin', 'uracil', 'ethanol', 'benzene']

config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for mol in molecules2:
    
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --eval_model=post"
    subprocess.run(command.split(" "))



