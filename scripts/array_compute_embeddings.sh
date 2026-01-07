#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200G
#SBATCH -p batch
#SBATCH --array=1-8
#SBATCH -J procupine
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --mail-user=maxwell_adorsoo@brown.edu
#SBATCH --mail-type=ALL

QUERY_PATH="/users/madorsoo/scratch/GenePT/src/query_list.txt"


query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)
INPUT_CSV="/users/madorsoo/scratch/GenePT/summaries/${query_name}/summaries.csv"
DATA_PATH="/users/madorsoo/scratch/GenePT/summary_dataset/${query_name}.h5ad"
OUTPUT_CSV="/users/madorsoo/scratch/GenePT/training_data/${query_name}.csv"
BIOGPT_EMBEDDINGS="/users/madorsoo/scratch/GenePT/gene_embeddings/biogpt_embeddings.pickle"



echo "processing ${query_name}"


conda run -n procupine python /users/madorsoo/scratch/GenePT/src/compute_embedding.py \
    --adata_path ${DATA_PATH} \
    --input_csv ${INPUT_CSV} \
    --output_csv ${OUTPUT_CSV} \
    --biogpt_embeddings ${BIOGPT_EMBEDDINGS} \
# removed vocab file option : --vocab-file ${VOCAB_PATH} \

