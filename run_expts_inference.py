import subprocess
from itertools import product


molecules1 = ['naphthalene', 'salicylic_acid', 'toluene']
molecules2 = ['aspirin', 'uracil', 'ethanol', 'benzene']

config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for mol in molecules2:
    
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --eval_model='50k'"
    subprocess.run(command.split(" "))

    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --eval_model='10k'"
    subprocess.run(command.split(" "))


