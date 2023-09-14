import subprocess
from itertools import product


molecules1 = ['DHA', 'double_walled_nanotube', 'buckyball_catcher']
molecules2 = ['stachyose', 'AT_AT_CG_CG', 'AT_AT', 'ac_Ala3_NHMe']


config_yml = '/home/sanjeevr/implicit-md/configs/md17/base.yml'
for mol in molecules1:
    command = f"python nve_implicit.py --config-yml {config_yml} \
                 --molecule={mol}"
                
    subprocess.run(command.split(" "))


