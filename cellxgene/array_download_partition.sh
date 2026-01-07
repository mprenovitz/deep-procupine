#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p batch 
#SBATCH --array=1-9
#SBATCH -J procupine
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --mail-user=maxwell_adorsoo@brown.edu
#SBATCH --mail-type=ALL

module load miniconda3/23.11.0s-odstpk5

INDEX_PATH="/users/madorsoo/scratch/procupine/scGPT/data/indices"
QUERY_PATH="/users/madorsoo/scratch/procupine/scGPT/data/cellxgene/query_list.txt"
DATA_PATH="/users/madorsoo/scratch/procupine/scGPT/data/datasets"


query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "downloading ${query_name}"


/users/madorsoo/scratch/procupine/scGPT/data/cellxgene/download_partition.sh ${query_name} ${INDEX_PATH} ${DATA_PATH}

