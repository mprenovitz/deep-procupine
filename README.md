# Abstract


# Preprocessing datasets

1. Follow instructions this [readme file](./cellxgene/README.md) to download datasets. and use the [preprocess.py](./src/preprocess.py) script for data preprocessing.

2. We used this [script](./src/request_ncbi_text_for_genes.ipynb) to download gene summaries. However, gene summaries can be obtained from any reliable database of one's choice. Here is a sample gene [summary](./gene_summaries/sample.json).

3. The summaries obtained above were used to get BioGPT embeddings using this other [script](./src/gene_embeddings_from_summaries.py). We also obtained nomic_ai embeddings for comparison, but couldn't do that for the project due time constraint. Pickle files of embeddings can be found in `gene_embeddings` directory.

4. The the ground truth summaries for augmenting ProCyon consist of Uniprot cell summaries and single cell dataset information. This was built from this [script](./src/build_procyon_cell_summaries.py)

5. The embeddings obtained in (3) are for all the genes in our corpus. These embedding are then mapped to top genes for each tissue in our dataset using [compute_embeddings](./src/compute_embeddings.py) script.

6. The final data format for the training pipeline is consolidated csv file containing relevant fields. A sample of the format for the pbmc dataset is provided [here](./training_data/pbmc.csv).