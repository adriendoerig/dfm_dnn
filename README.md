### How to run on bluebear gpus

```
ssh bluebear.bham.ac.uk
module load slurm-interactive
fisbatch_screen --nodes 1-1 --ntasks 12 --mem=32G --time 72:0:0 --qos=bbgpu --account=charesti-start
source tf_env_activate.sh
cd /castles/nr/projects/2018/charesti-start/projects/dfm_dnn
```

then you can run the code using python like usual. 