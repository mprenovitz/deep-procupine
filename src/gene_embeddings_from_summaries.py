"""
Get Gene embedding form gene summaries. Use either nomic-ai text embedding or biogpt embedding.
Works with either csv or json file. We assume that first line in csv file is a header, and that json files are in the format expected by pandas' read_json.
Tweak the main code to suit your specific file structure.
"""
from transformers import BioGptTokenizer, BioGptModel
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from typing import List
import pickle
import argparse
import os
import sys

sys.path.insert(0, "../../")


def get_biogpt_embedding_wrapper():
    """
    Return a funciton that get BioGPT embeddings from gene_summaries.

    Args:
        gene_summary: text to embed
    
    Returns:
        Returns a ndarry of embedding vector of shape (1024, )
    """
    # Load model and tokenizer
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt")

    def get_embedding(gene_summary:str):
        # tokenize the input
        inputs = tokenizer(gene_summary, return_tensors="pt", truncation=True, max_length=1024)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use embedding mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy()
    
    return get_embedding



def get_nomic_ai_embeddings(gene_summaries: List[str]):
    """
    Return a 2D numpy embedding vectors from a list of texts.

    Args:
        texts:  list of text
    
    Notes:
        The List of text should have 'search_document:' prepended to the actual text.
        e.g: 
        # For gene summaries, genes summaries will be prefixed with the 'search_document'
        >> gene_summaries = [
        'search_document: TP53 encodes tumor protein p53...',
        'search_document: BRCA1 is a tumor suppressor gene...'
        ]
    
    Returns:
        A 2D numpy array of embeddeding vecors of dimension (len(text), 768)
    """
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    embeddings = model.encode(gene_summaries)
    print(f"Successfully generate embeddings of shape {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build gene summaries from NCBI summaries page"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path for input file containing gene summaries. Should be csv or json file"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        default="csv",
        help="Format of file. Either csv or json. default (csv)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to store embedding in a pickle file. If not provided will created outside the parent directory of script files"
    )

    parser.add_argument(
       "--model",
       type=str,
       required=False,
       default="nomic",
       choices=["nomic", "biogpt"],
       help="Model used for embedding. Chose from (nomic or biogpt); default (nomic)" 
    )

    parser.add_argument(
        "--colname",
        type=str,
        required=False,
        default="summary",
        help="Column name for the table column containing summaries"
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    file_type = args.file
    model_type = args.model
    header = args.colname

    # Read summaries file
    reader = pd.read_csv if file_type == "csv" else pd.read_json
    summaries_df = None # Dummy initialization
    try:
        summaries_df = reader(input_path)
        l0 = summaries_df.shape[0]
        summaries_df = summaries_df[summaries_df['summary'] != ""]
        l1 = summaries_df.shape[0]
        print(f"{l1 - l0} empty summary enteries removed.")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Exiting Program ...")
        exit(1)

    # Get embeddings 
    try:
        embeddings = []
        if model_type == "biogpt":
            get_embedding = get_biogpt_embedding_wrapper()
            embeddings = summaries_df[header].apply(get_embedding)
            embeddings = embeddings.to_numpy()
        elif model_type == "nomic":
            headers = "search_document: " + summaries_df[header].astype(str)
            print(f"sending .. {headers.head()}")
            embeddings = get_nomic_ai_embeddings(list(summaries_df[header]))
            keys = [summaries_df]
    except Exception as e:
        print(f"Error getting gene embeddings: {e}")
        print("Exiting Program ...")
        exit(1)
    
    # Save embeddings to file
    try:
        if output_dir is None:
            output_dir = os.path.join(os.path.split(os.getcwd())[0], f"gene_embeddings")
        filepath = os.path.join(output_dir, f"{model_type}_embeddings.pickle")

        os.makedirs(output_dir, exist_ok=True)

        print(f"{len(embeddings)} {model_type} embeddings generated")
        embeddings_dict = dict(zip(summaries_df["gene"], embeddings))
        
        with open(filepath, "wb") as f:
            pickle.dump(embeddings_dict, f)
        print(f"Successfully saved embeddings to {filepath}")
    except Exception as e:
        print(f"Error saving embeddings to file: {e}")