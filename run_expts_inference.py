import subprocess
from itertools import product

command = f"python nvt_implicit.py --config-yml /home/sanjeevr/implicit-md/configs/water/simulate.yml --eval_model=pre --energy_force_loss_weight=0"
subprocess.run(command.split(" "))
for i in range(4):
    command = f"python nvt_implicit.py --config-yml /home/sanjeevr/implicit-md/configs/water/simulate.yml --eval_model=post_cycle{i+1} --energy_force_loss_weight=0"
    subprocess.run(command.split(" "))