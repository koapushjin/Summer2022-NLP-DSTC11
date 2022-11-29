check:
	squeue -u ${USER} -l

clean:
	rm -f *.err *.out

train:
	sbatch --partition=aquila ./run.sh

rtrain:
	sbatch --partition=aquila ./run_roberta.sh

tmp:
	srun --pty --jobid <Job_ID> /usr/bin/bash

del:
	scancel <Job_ID>
