check:
	squeue -u ${USER} -l

clean:
	rm -f log_train_*.err log_train_*.out

tocd:
	sbatch --partition=aquila ./run-tocd.sh

cat:
	sbatch --partition=aquila ./run-tocd-cat.sh
testt:
	sbatch --partition=aquila ./run-test-tocd.sh

mtocd:
	sbatch --partition=aquila ./run-mask-tocd.sh

tmp:
	srun --pty --jobid <Job_ID> /usr/bin/bash

del:
	scancel <Job_ID>

tocde:
	sbatch --partition=aquila ./run-train-esb.sh

teste:
	sbatch --partition=aquila ./run-test-esb.sh
