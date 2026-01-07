import numpy as np
import pandas as pd
import pickle
import scanpy as sc
from scipy import sparse
from pathlib import Path
import argparse
from tqdm import tqdm


def standardize_expression(expr_values):
    """
    Standardize expression values to zero mean and unit variance.
    
    Args:
        expr_values: Array of expression values
    
    Returns:
        Standardized array
    """
    if len(expr_values) == 0 or expr_values.std() == 0:
        return expr_values
    
    # Standardize: (x - mean) / std
    mean = expr_values.mean()
    std = expr_values.std()
    return (expr_values - mean) / std


def process_single_gene(gene_name, X_data, var_data, emb_dim, feature_col, gene_to_idx):
    """
    Process a single gene to get top N cell expression values.
    N matches the embedding dimension for proper matrix multiplication.
    """
    gene_idx = gene_to_idx.get(gene_name)
    
    if gene_idx is None:
        return None
    
    # Get expression for this gene across all cells
    gene_expr = X_data[:, gene_idx]
    
    if sparse.issparse(gene_expr):
        gene_expr = gene_expr.toarray().flatten()
    else:
        gene_expr = np.asarray(gene_expr).flatten()
    
    # Get top emb_dim cells (so dimensions match for matrix mult)
    top_n = min(emb_dim, len(gene_expr))
    top_cell_indices = np.argsort(gene_expr)[-top_n:][::-1]
    top_cell_values = gene_expr[top_cell_indices]
    
    # Pad with zeros if we don't have enough cells
    if top_n < emb_dim:
        top_cell_values = np.pad(top_cell_values, (0, emb_dim - top_n), 
                                 mode='constant', constant_values=0)
    
    # Standardize to zero mean, unit variance
    top_cell_values_std = standardize_expression(top_cell_values)
    
    return top_cell_values_std


def compute_top_expressions_sequential(adata, gene_names_list, emb_dim, feature_col=None):
    """
    Get top N cell expression values for each gene sequentially.
    Standardizes expressions to zero mean, unit variance.
    
    Args:
        adata: AnnData object
        gene_names_list: List of gene names from CSV
        emb_dim: Embedding dimension (number of top cells to extract)
        feature_col: Column containing gene names
    
    Returns:
        Dictionary mapping gene names to standardized top expression arrays
    """
    print(f"Extracting top {emb_dim} cells for {len(gene_names_list)} genes")
    print(f"Will standardize to zero mean, unit variance")
    
    # Build mapping from gene name to column index
    if feature_col and feature_col in adata.var.columns:
        all_gene_names = adata.var[feature_col].values
    else:
        all_gene_names = adata.var.index.values
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_gene_names)}
    
    # Process genes sequentially with progress bar
    results = {}
    for gene_name in tqdm(gene_names_list, desc="Processing genes"):
        result = process_single_gene(
            gene_name, 
            adata.X, 
            adata.var, 
            emb_dim, 
            feature_col, 
            gene_to_idx
        )
        results[gene_name] = result
    
    print(f"Extracted standardized expression for {len(results)} genes")
    
    return results


def compute_weighted_embeddings(top_expressions_dict, gene_embeddings_dict, gene_names_list):
    """
    Compute weighted embeddings via element-wise multiplication.
    
    For each gene:
        top_expr: (emb_dim,) - standardized top cell expressions
        gene_emb: (emb_dim,) - gene embedding
        weighted = top_expr * gene_emb (element-wise)
        result: (emb_dim,) - weighted embedding
    
    Args:
        top_expressions_dict: {gene_name: standardized top N expression array}
        gene_embeddings_dict: {gene_name: embedding array}
        gene_names_list: Ordered list of genes
    
    Returns:
        Dictionary of {gene_name: weighted_embedding_string}
    """
    results = {}
    
    for gene_name in tqdm(gene_names_list, desc="Computing weighted embeddings"):
        top_expr = top_expressions_dict.get(gene_name)
        gene_emb = gene_embeddings_dict.get(gene_name)
        
        if top_expr is None:
            results[gene_name] = ''
            continue
            
        if gene_emb is None:
            results[gene_name] = ''
            continue
        
        # Ensure dimensions match
        if len(top_expr) != len(gene_emb):
            results[gene_name] = ''
            continue
        
        # Element-wise multiplication: expression weights embedding
        weighted_embedding = top_expr * gene_emb  # (emb_dim,)
        
        # Convert to string
        embedding_str = ','.join(map(str, weighted_embedding))
        results[gene_name] = embedding_str
    
    return results


def detect_feature_column(adata):
    """Auto-detect the column containing gene symbols."""
    SYMBOL_KEYS = ['feature_name', 'gene_symbols', 'gene_name', 'Symbol']
    
    for key in SYMBOL_KEYS:
        if key in adata.var.columns:
            print(f"Detected gene symbol column: {key}")
            return key
    
    print("No gene symbol column found, using var_names")
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute weighted gene embeddings from scRNA-seq data')
    parser.add_argument('--adata_path', type=str, required=True, help='Path to AnnData file')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV with gene_name column')
    parser.add_argument('--biogpt_embeddings', type=str, default=None, help='Path to BioGPT embeddings pickle')
    parser.add_argument('--nomic_embeddings', type=str, default=None, help='Path to Nomic embeddings pickle')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV')
    parser.add_argument('--feature_col', type=str, default=None, help='var column containing genes')
    
    args = parser.parse_args()
    
    print("="*80)
    print("WEIGHTED GENE EMBEDDING COMPUTATION")
    print("="*80)
    
    # Load input CSV
    print("\n1. Loading input CSV...")
    input_df = pd.read_csv(args.input_csv)
    if 'gene_name' not in input_df.columns:
        raise ValueError("Input CSV must have 'gene_name' column")
    
    gene_names_list = input_df['gene_name'].tolist()
    print(f"   Found {len(gene_names_list)} genes in CSV")
    print(f"   Example genes: {gene_names_list[:5]}")
    
    # Load AnnData
    print("\n2. Loading AnnData...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"   Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Detect feature column
    if args.feature_col is None:
        args.feature_col = detect_feature_column(adata)
    
    # Process BioGPT
    if args.biogpt_embeddings:
        print("\n3. Processing BioGPT embeddings (1024-dim)...")
        
        # Step 1: Get top 1024 expression values (standardized)
        print("   Step 1: Extracting & standardizing top 1024 cell expressions...")
        top_expr_1024 = compute_top_expressions_sequential(
            adata=adata,
            gene_names_list=gene_names_list,
            emb_dim=1024,
            feature_col=args.feature_col
        )
        
        # Step 2: Load embeddings
        print("   Step 2: Loading BioGPT embeddings...")
        with open(args.biogpt_embeddings, "rb") as f:
            biogpt_embeddings = pickle.load(f)
        print(f"   Loaded {len(biogpt_embeddings)} gene embeddings")
        
        # Step 3: Compute weighted embeddings
        print("   Step 3: Computing weighted embeddings (expr * emb)...")
        biogpt_dict = compute_weighted_embeddings(
            top_expr_1024,
            biogpt_embeddings,
            gene_names_list
        )
        
        # Add to dataframe
        biogpt_zero = ','.join(['0.0'] * 1024)
        input_df['biogpt_embedding'] = input_df['gene_name'].map(biogpt_dict).fillna(biogpt_zero)
        print(f"   ✓ Added BioGPT weighted embeddings")
    
    # Process Nomic
    if args.nomic_embeddings:
        print("\n4. Processing Nomic embeddings (768-dim)...")
        
        # Step 1: Get top 768 expression values (standardized)
        print("   Step 1: Extracting & standardizing top 768 cell expressions...")
        top_expr_768 = compute_top_expressions_sequential(
            adata=adata,
            gene_names_list=gene_names_list,
            emb_dim=768,
            feature_col=args.feature_col
        )
        
        # Step 2: Load embeddings
        print("   Step 2: Loading Nomic embeddings...")
        with open(args.nomic_embeddings, 'rb') as f:
            nomic_embeddings = pickle.load(f)
        print(f"   Loaded {len(nomic_embeddings)} gene embeddings")
        
        # Step 3: Compute weighted embeddings
        print("   Step 3: Computing weighted embeddings (expr * emb)...")
        nomic_dict = compute_weighted_embeddings(
            top_expr_768,
            nomic_embeddings,
            gene_names_list
        )
        
        # Add to dataframe
        nomic_zero = ','.join(['0.0'] * 768)
        input_df['nomic_embedding'] = input_df['gene_name'].map(nomic_dict).fillna(nomic_zero)
        print(f"   ✓ Added Nomic weighted embeddings")
    
    # Save
    print("\n5. Saving results...")
    input_df.to_csv(args.output_csv, index=False)
    print(f"   ✓ Saved to {args.output_csv}")
    print(f"   ✓ Final shape: {input_df.shape}")
    
    # Show computation summary
    print("\n" + "="*80)
    print("COMPUTATION SUMMARY")
    print("="*80)
    print("For each gene:")
    print("  1. Extract top N cells (N = embedding dimension)")
    print("  2. Standardize expression: (x - mean) / std")
    print("  3. Element-wise multiply: weighted[i] = standardized_expr[i] * embedding[i]")
    print()
    if args.biogpt_embeddings:
        print("BioGPT: top 1024 cells × 1024-dim embedding → 1024-dim weighted embedding")
    if args.nomic_embeddings:
        print("Nomic:  top 768 cells × 768-dim embedding → 768-dim weighted embedding")
    print("="*80)


if __name__ == "__main__":
    main()