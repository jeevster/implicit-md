import subprocess
from itertools import product


molecules1 = ['aspirin', 'naphthalene', 'salicylic_acid']
molecules2 = ['ethanol', 'toluene', 'malonaldehyde']
molecules3 = ['benzene', 'uracil']
optimizers = ['Adam', 'SGD']
steps = [1000]
mse_statuses = [True, False]
allow_off_policy_updates = [False]
config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for element in product(molecules3, optimizers, mse_statuses, allow_off_policy_updates):
    mol, opt, mse, allow = element
    name = f"dynamics100_{opt}_usemsegradient_{mse}_allowoffpolicy_{allow}"
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol} --optimizer={opt} \
                 --use_mse_gradient={mse} --allow_off_policy_updates={allow} \
                     --exp_name={name}"
                
    subprocess.run(command.split(" "))


