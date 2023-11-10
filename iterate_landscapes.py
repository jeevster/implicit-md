from subprocess import run

distances = [0.1, 0.2, 0.3, 0.4, 0.5]

for dist in distances:
    command = f"python /home/sanjeevr/implicit-md/ip_explorer/ip_explorer/landscape/__main__.py --num-nodes=1 --save-dir /home/sanjeevr/implicit-md/landscapes --model-type=nequip --database-path=/home/sanjeevr/MDsim/MODELPATH/nequip/md22-ac_Ala3_NHMe_5percent_lmax=1_nequip --model-path=/home/sanjeevr/MDsim/MODELPATH/nequip/md22-ac_Ala3_NHMe_5percent_lmax=1_nequip --landscape-type=plane --no-compute-initial-losses --steps=10 --distance={dist}"
    run(command.split(" "))

