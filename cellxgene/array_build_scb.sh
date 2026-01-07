#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH -p batch
#SBATCH --array=1-9
#SBATCH -J procupine
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --mail-user=maxwell_adorsoo@brown.edu
#SBATCH --mail-type=ALL

QUERY_PATH="/users/madorsoo/scratch/procupine/scGPT/data/cellxgene/query_list.txt"


query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

DATA_PATH="/users/madorsoo/scratch/procupine/scGPT/data/datasets/${query_name}"
OUTPUT_PATH="/users/madorsoo/scratch/procupine/scGPT/data/preprocessed/${query_name}"
#VOCAB_PATH="path/to/vocab"

echo "processing ${query_name}"
N=200000


mkdir -p $OUTPUT_PATH

echo "downloading to ${OUTPUT_PATH}" 

conda run -n procupine python build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --N ${N}

# removed vocab file option : --vocab-file ${VOCAB_PATH} \