module load NiaEnv/2019b python/3.8.5
source ~/.virtualenvs/env/bin/activate
module load gcc
export PYTHONPATH="${PYTHONPATH}:/home/r/rbond/gallardo/projects/mond/fit_powerlaws/commonlib"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
