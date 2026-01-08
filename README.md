# Abstract

# Preprocessing datasets - Maxwell

1. Follow instructions this [readme file](./cellxgene/README.md) to download datasets. and use the [preprocess.py](./src/preprocess.py) script for data preprocessing.

2. We used this [script](./src/request_ncbi_text_for_genes.ipynb) to download gene summaries. However, gene summaries can be obtained from any reliable database of one's choice. Here is a sample gene [summary](./gene_summaries/sample.json).

3. The summaries obtained above were used to get BioGPT embeddings using this other [script](./src/gene_embeddings_from_summaries.py).

4. The the ground truth summaries for augmenting ProCyon consist of Uniprot cell summaries and single cell dataset information. This was built from this [script](./src/build_procyon_cell_summaries.py)

5. The embeddings obtained in (3) are for all the genes in our corpus. These embedding are then mapped to top genes for each tissue in our dataset using [compute_embeddings](./src/compute_embeddings.py) script.

6. The final data format for the training pipeline is consolidated csv file containing relevant fields. A sample of the format for the pbmc dataset is provided [here](./training_data/pbmc.csv).

# Training and Inference - Dhruv

1. Obtained ProCyon pretrained weights and Llama weights and split them onto two GPUs

2. Preprocess the BioGPT data, mapping UniProt IDs to ProCyon indices, filtering the dataset.

3. Align gene embeddings to gene summaries (text), tokenize text, and convert gene features to padded torch tensors. I've also added attention masks to to the sequences.

4. Train an adapter MLP to map protein and gene embeddings to a shared embedding space-- text encoder and gene adapter are on one GPU while the protein branch stays on the other. For this task, the ground truth is the gene summaries (we want the protein information to be gene-aware). The gene embedding is projected by the adapter, concatenated with the protein and text embeddings, and then run through all layers of the frozen Llama model. We use gradient accumulation with a batch size of 1 or 2 to prevent memory management issues.

5. Generate phenotype candidates using ProCyon's input structure, injecting the gene embedding into the prompt itself (which already has protein embeddings)

6. For given proteins, run QA and score candidate texts by calculating the next-token-probability of "Yes" for our model and ProCyon's model for comparison.

# Benchmarking and Evaluation - Matt

1. Takes in generated Protein Phenotypes and Ground truth values to calculate BERTScore, with optional bootstrapped 95 confidence interval, as well as an LLM-as-judge approach for biological relevance of phenotype.

2. BERTScore provides an accurate and flexible semantic similarity score to the ground truth because it takes into acount word context embeddings rather than exact matches to be more robust towards varying descriptions that may capture the same meaning.

3. The LLM-as-judge approach is used to assess how coherent and biologically accurate described phenotypes are when compared to the ground truth. This approach compares the ProCyon phenotype description with the Procupine equivalent and determines which is "better" compared to the ground truth with ties being possible.

4. To match the benchmarking carried out in ProCyon we used SciBert for the BertScorer, but did not have access to Anthropics Sonnet 3.5 at the time of completion for our LLM-as-judge, so Qwen-2.5 instruct was used as it fit within our computational constraints. 

5. When determining BERTScore and LLM-as-judge results, use the generated phenotype with the highest confidence interval.

6. Note that for comparison between generated phenotypes of a single protein the bootstrapped confidence interval is unecessary and should only be used for large quantities of data for statistical signifigance. 
