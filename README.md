# How to run on bluebear gpus

## turns out that the virtual env is slowing tf down like hell

### here's a way to get training done on a gpu in a fraction of the time.

```
ssh bluebear.bham.ac.uk
module load slurm-interactive
fisbatch_screen --nodes 1-1 --ntasks 20 --mem=32G --time 72:0:0 --qos=castlespowergpu --account=charesti-start --gres=gpu:v100:1
module load bear-apps/2019b
module load TensorFlow/2.0.0-fosscuda-2019b-Python-3.7.4
module load IPython/7.9.0-fosscuda-2019b-Python-3.7.4
cd /castles/nr/projects/2018/charesti-start/projects/dfm_dnn
```

then you can run the code using python like usual. 
