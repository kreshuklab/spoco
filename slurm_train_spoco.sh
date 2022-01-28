#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=adrian.wolny@hhi-extern.fraunhofer.de
#SBATCH --job-name=spoco
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --mem=64G

exit_code=0

# include the definition of the LOCAL_JOB_DIR which is autoremoved after each job
source "/etc/slurm/local_job_dir.sh"

# create results dir where checkpoints are saved
mkdir -p "${LOCAL_JOB_DIR}/job_results"

# copy the compressed dataset to $LOCAL_JOB_DIR which is created for each individual job locally on the node where the job or steps of it are running
cp ${SLURM_SUBMIT_DIR}/Cityscapes.zip ${LOCAL_JOB_DIR}

# create the directory dataset before uncompressing into it
unzip ${LOCAL_JOB_DIR}/Cityscapes.zip -d ${LOCAL_JOB_DIR}

# launch the singularity image and bind $LOCAL_JOB_DIR on this node to /mnt as used within the singularity image
srun singularity run --nv --bind ${LOCAL_JOB_DIR}:/mnt ./train.sif $@
ret_val=$?; if (( $ret_val > $exit_code )); then exit_code=$ret_val; fi

# save job results
cd "$LOCAL_JOB_DIR"
tar -cf zz_${SLURM_JOB_ID}.tar job_results
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR
rm -rf ${LOCAL_JOB_DIR}/job_results

exit $exit_code

# run CL with Dice and Consistency
# sbatch -p gpu1or3 slurm_train_spoco.sh --spoco --ds-name cityscapes --ds-path /mnt/Cityscapes --things-class bicycle --batch-size 16 --loss-unlabeled-push 0.0 --checkpoint-dir /mnt/job_results --log-after-iters 2000 --max-num-iterations 90000 --max-num-validations 100 --cos
