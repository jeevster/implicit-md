import subprocess
from itertools import product


molecules1 = ['aspirin', 'naphthalene', 'salicylic_acid']
molecules2 = ['ethanol', 'toluene']
molecules3 = ['benzene', 'uracil']


config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for mol in molecules3:
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol}"
                
    subprocess.run(command.split(" "))


