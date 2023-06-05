check:
	squeue -u ${USER} -l

clean:
	rm -f log_train_*.err log_train_*.out

xvalird:
	sbatch --partition=aquila ./run_xvali.sh
