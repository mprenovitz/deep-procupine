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
PROTEIN_CACHE="/users/madorsoo/scratch/GenePT/summaries/${query_name}/protein_cache.pkl"
DATA_PATH="/users/madorsoo/scratch/GenePT/summary_dataset/${query_name}.h5ad"

echo "processing ${query_name}"


conda run -n procupine python /users/madorsoo/scratch/GenePT/src/generate_gene_summaries.py \
    --adata_path ${DATA_PATH} \
    --tissue ${query_name} \
    --protein_cache_path ${PROTEIN_CACHE} \
# removed vocab file option : --vocab-file ${VOCAB_PATH} \

