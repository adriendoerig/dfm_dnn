# This script is intended to be sourced. It will set up a virtualenv
# if one doesn't already exist, and leave you ready to run the scripts
# in this folder.

module load bear-apps/2019b
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.0.0-fosscuda-2019b-Python-3.7.4


VENV="env-tf-${BB_OSBASE}-${BB_CPU}"

if ! [[ -e ${VENV} ]]; then
   virtualenv --python ${EBROOTPYTHON}/bin/python ${VENV}
fi

source ${VENV}/bin/activate

${VENV}/bin/pip install -r requirements.txt
