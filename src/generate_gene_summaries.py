import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import random
import argparse
import os
from pathlib import Path
import scanpy as sc
import pickle
from tqdm import tqdm


class GeneSummaryGenerator:
    """Generate rich, ProCyon-style instructional summaries for genes with cellular context."""
    
    # Sentence templates for variation
    INTRO_TEMPLATES = [
        "The gene {gene_name} encodes {protein_name} and is expressed across {n_cells:,} cells in this dataset. ",
        "{gene_name}, which encodes {protein_name}, shows detectable expression in {n_cells:,} cells. ",
        "Expression analysis reveals that {gene_name} ({protein_name}) is active in {n_cells:,} cells. ",
        "The {gene_name} gene, encoding {protein_name}, is transcribed in {n_cells:,} cells within this sample. ",
        "{gene_name} encodes {protein_name} and demonstrates expression in {n_cells:,} cells. "
    ]
    
    EXPRESSION_TEMPLATES = [
        "Expression is highest in {top_cell_types}, with mean expression of {mean_expr:.2f} and peak expression reaching {max_expr:.2f}. ",
        "The gene shows preferential expression in {top_cell_types}, averaging {mean_expr:.2f} with maximum levels of {max_expr:.2f}. ",
        "Top expressing cell populations include {top_cell_types}, with average expression {mean_expr:.2f} and peak values of {max_expr:.2f}. ",
        "Prominent expression is observed in {top_cell_types}, displaying mean levels of {mean_expr:.2f} and maximal expression of {max_expr:.2f}. ",
        "Cell type-specific expression peaks in {top_cell_types}, with mean expression {mean_expr:.2f} and maximum {max_expr:.2f}. "
    ]
    
    PROTEIN_FUNCTION_TEMPLATES = [
        "At the protein level, {protein_name} functions as follows: {function} ",
        "The encoded protein, {protein_name}, {function} ",
        "Functionally, {protein_name} {function} ",
        "{protein_name} plays a biological role: {function} ",
        "The protein product {protein_name} {function} "
    ]
    
    CELL_CONTEXT_TEMPLATES = [
        "In the top {top_n:,} expressing cells, this gene is co-expressed with {coexpr_genes}, suggesting coordinated biological functions. ",
        "Among the {top_n:,} highest expressing cells, co-expression patterns include {coexpr_genes}, indicating functional relationships. ",
        "In high-expression contexts (top {top_n:,} cells), co-expression with {coexpr_genes} points to shared cellular programs. ",
        "The top {top_n:,} expressing cells show concurrent expression of {coexpr_genes}, reflecting coordinated regulation. ",
        "Among {top_n:,} cells with highest expression, {coexpr_genes} are co-expressed, suggesting participation in common biological processes. "
    ]
    
    def __init__(self, adata, protein_cache=None, feature_col=None, output_dir=None, save_frequency=100):
        self.adata = adata
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_frequency = save_frequency
        
        # Auto-detect feature column if not provided
        if feature_col is None:
            feature_col = self._detect_feature_column()
        self.feature_col = feature_col
        
        # Get gene names using the detected/specified column
        self.gene_names = self._get_gene_names()
        self.protein_cache = protein_cache if protein_cache else {}
        
        # Check for cell type annotations
        self.cell_type_col = self._find_cell_type_column()
        
        # Filter to HVGs only
        if 'highly_variable' not in adata.var.columns:
            raise ValueError("AnnData must have 'highly_variable' column in var. Run sc.pp.highly_variable_genes() first.")
        
        self.hvg_mask = adata.var['highly_variable'].values
        self.hvg_indices = np.where(self.hvg_mask)[0]
        self.hvg_names = self._get_gene_names()[self.hvg_mask]
        
        print(f"Found {len(self.hvg_names)} highly variable genes")
        print(f"Using feature column: {self.feature_col}")
        print(f"Example genes: {self.hvg_names[:5]}")
    
    def _detect_feature_column(self):
        """Auto-detect the column containing gene symbols."""
        SYMBOL_KEYS = ['feature_name', 'gene_symbols', "gene_name", "Symbol"]
        
        for key in SYMBOL_KEYS:
            if key in self.adata.var.columns:
                print(f"Detected gene symbol column: {key}")
                return key
        
        print("No gene symbol column found, using var_names")
        return None
    
    def _get_gene_names(self):
        """Get gene names/symbols as array."""
        if self.feature_col and self.feature_col in self.adata.var.columns:
            return self.adata.var[self.feature_col].values
        else:
            return self.adata.var_names.values
    
    def _find_cell_type_column(self):
        """Find cell type column in obs."""
        for col in ['cell_type', 'celltype', 'cell_type_name', 'leiden', 'louvain']:
            if col in self.adata.obs.columns:
                return col
        return None

    def get_protein_info(self, gene_name: str) -> Optional[Dict]:
        """Retrieve protein information from UniProt with retry logic."""
        if gene_name in self.protein_cache:
            return self.protein_cache[gene_name]

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': f'gene:{gene_name} AND organism_name:human',
            'format': 'json',
            'fields': 'accession,gene_names,protein_name,cc_function,cc_subcellular_location,go_p',
            'size': 1
        }

        retry_strategy = Retry(
            total=5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)
        
        try:
            response = http.get(url, params=params, timeout=50) 
            response.raise_for_status() 

            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                
                protein_name = gene_name 
                function = ''
                location = ''
                uniprot_id = result.get('primaryAccession', '')
                go_terms = []

                if 'proteinDescription' in result:
                    rec_name = result['proteinDescription'].get('recommendedName', {})
                    name_value = rec_name.get('fullName', {}).get('value')
                    if name_value:
                        protein_name = name_value

                if 'comments' in result:
                    for comment in result['comments']:
                        if comment.get('commentType') == 'FUNCTION':
                            texts = comment.get('texts', [])
                            if texts:
                                function = texts[0].get('value', '')
                            break

                    for comment in result['comments']:
                        if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                            subcell_locs = comment.get('subcellularLocations', [])
                            if subcell_locs:
                                loc_obj = subcell_locs[0].get('location', {})
                                location = loc_obj.get('value', '')
                            break
                
                if 'uniProtKBCrossReferences' in result:
                    for ref in result['uniProtKBCrossReferences']:
                        if ref.get('database') == 'GO':
                            go_terms.append(ref.get('id', ''))
                
                info = {
                    'gene': gene_name,
                    'protein_name': protein_name,
                    'function': function,
                    'location': location,
                    'uniprot_id': uniprot_id,
                    'go_terms': go_terms[:5]
                }
                
                self.protein_cache[gene_name] = info
                return info

        except Exception as e:
            pass

        return {
            'gene': gene_name,
            'protein_name': gene_name,
            'function': '',
            'location': '',
            'uniprot_id': '',
            'go_terms': []
        }
    
    def get_top_expressing_cells(self, gene_idx: int, top_n: int = 100) -> np.ndarray:
        """Get top N expressing cells for a gene."""
        gene_expr = self.adata.X[:, gene_idx]
        
        if hasattr(gene_expr, 'toarray'):
            gene_expr = gene_expr.toarray().flatten()
        else:
            gene_expr = np.asarray(gene_expr).flatten()
        
        top_cell_indices = np.argsort(gene_expr)[-top_n:][::-1]
        return top_cell_indices
    
    def get_coexpressed_genes(self, gene_idx: int, top_cell_indices: np.ndarray, top_n: int = 5) -> List[str]:
        """Get genes co-expressed in the top cells."""
        if hasattr(self.adata.X, 'toarray'):
            top_cell_expr = self.adata.X[top_cell_indices, :].toarray()
        else:
            top_cell_expr = self.adata.X[top_cell_indices, :]
        
        mean_expr = top_cell_expr.mean(axis=0)
        if hasattr(mean_expr, 'A1'):
            mean_expr = mean_expr.A1
        else:
            mean_expr = np.asarray(mean_expr).flatten()
        
        top_indices = np.argsort(mean_expr)[::-1]
        coexpr_genes = []
        for idx in top_indices:
            if idx != gene_idx and len(coexpr_genes) < top_n:
                coexpr_genes.append(self.gene_names[idx])
        
        return coexpr_genes
    
    def get_top_cell_types(self, top_cell_indices: np.ndarray, top_n: int = 3) -> List[str]:
        """Get cell types most represented in top expressing cells."""
        if self.cell_type_col is None:
            return []
        
        cell_types = self.adata.obs[self.cell_type_col].values[top_cell_indices]
        unique, counts = np.unique(cell_types, return_counts=True)
        
        sorted_indices = np.argsort(counts)[::-1]
        top_types = [unique[i] for i in sorted_indices[:top_n]]
        
        return top_types
    
    def generate_gene_summary(self, gene_name: str, gene_idx: int, 
                             top_n_cells: int = 100, seed: Optional[int] = None) -> str:
        """Generate a rich, instructional summary for a gene."""
        if seed is not None:
            random.seed(seed)
        
        protein_info = self.get_protein_info(gene_name)
        
        gene_expr = self.adata.X[:, gene_idx]
        if hasattr(gene_expr, 'toarray'):
            gene_expr = gene_expr.toarray().flatten()
        else:
            gene_expr = np.asarray(gene_expr).flatten()
        
        n_expressing = (gene_expr > 0).sum()
        mean_expr = gene_expr.mean()
        max_expr = gene_expr.max()
        
        top_cell_indices = self.get_top_expressing_cells(gene_idx, top_n_cells)
        coexpr_genes = self.get_coexpressed_genes(gene_idx, top_cell_indices, top_n=5)
        top_cell_types = self.get_top_cell_types(top_cell_indices, top_n=3)
        
        summary_parts = []
        
        intro = random.choice(self.INTRO_TEMPLATES).format(
            gene_name=gene_name,
            protein_name=protein_info['protein_name'],
            n_cells=n_expressing
        )
        summary_parts.append(intro)
        
        if top_cell_types:
            cell_type_str = ", ".join(top_cell_types)
        else:
            cell_type_str = "various cell populations"
        
        expr = random.choice(self.EXPRESSION_TEMPLATES).format(
            top_cell_types=cell_type_str,
            mean_expr=mean_expr,
            max_expr=max_expr
        )
        summary_parts.append(expr)
        
        if protein_info['function']:
            func_text = protein_info['function'].split('.')[0] + '.'
            if func_text and not func_text[0].isupper():
                func_text = func_text[0].lower() + func_text[1:]
            
            protein_func = random.choice(self.PROTEIN_FUNCTION_TEMPLATES).format(
                protein_name=protein_info['protein_name'],
                function=func_text
            )
            summary_parts.append(protein_func)
        
        if protein_info['location']:
            summary_parts.append(
                f"The protein localizes to the {protein_info['location'].lower()}. "
            )
        
        if coexpr_genes:
            coexpr_str = ", ".join(coexpr_genes)
            cell_context = random.choice(self.CELL_CONTEXT_TEMPLATES).format(
                top_n=top_n_cells,
                coexpr_genes=coexpr_str
            )
            summary_parts.append(cell_context)
        
        if protein_info['uniprot_id']:
            summary_parts.append(f"UniProt ID: {protein_info['uniprot_id']}.")
        
        return "".join(summary_parts)


def generate_all_gene_summaries_sequential(adata, top_n_cells: int = 100, seed: int = 42,
                                          feature_col: Optional[str] = None,
                                          output_dir: Optional[Path] = None,
                                          protein_cache_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate ProCyon-style summaries for highly variable genes sequentially.
    """
    
    # Auto-detect feature column
    if feature_col is None:
        SYMBOL_KEYS = ['feature_name', 'gene_symbols', 'gene_name', 'Symbol']
        for key in SYMBOL_KEYS:
            if key in adata.var.columns:
                feature_col = key
                break
    
    print(f"Using feature column: {feature_col if feature_col else 'var_names'}")
    
    # Check for HVGs
    if 'highly_variable' not in adata.var.columns:
        raise ValueError("AnnData must have 'highly_variable' column.")
    
    # Get HVG indices and names
    hvg_mask = adata.var['highly_variable'].values
    hvg_indices = np.where(hvg_mask)[0]
    
    if feature_col and feature_col in adata.var.columns:
        hvg_names = adata.var[feature_col][hvg_mask].values
    else:
        hvg_names = adata.var_names[hvg_mask].values
    
    print(f"Processing {len(hvg_names)} highly variable genes")
    print(f"Example genes: {hvg_names[:5]}")
    print(f"Top {top_n_cells} cells per gene")
    
    # Load protein cache
    if protein_cache_path is None:
        protein_cache_path = output_dir / "protein_cache.pkl"
    
    if os.path.exists(protein_cache_path):
        try:
            with open(protein_cache_path, 'rb') as f:
                protein_cache = pickle.load(f)
            print(f"Loaded existing protein cache with {len(protein_cache)} entries")
        except Exception as e:
            print(f"Error loading protein cache: {e}")
            protein_cache = {}
    else:
        protein_cache = {}
    
    # Create generator
    generator = GeneSummaryGenerator(
        adata,
        protein_cache=protein_cache,
        feature_col=feature_col,
        output_dir=output_dir
    )
    
    # Process genes sequentially with progress bar
    results = []
    save_frequency = 100
    
    for i, (gene_name, gene_idx) in enumerate(tqdm(zip(hvg_names, hvg_indices), 
                                                    total=len(hvg_names),
                                                    desc="Processing genes"), 1):
        gene_seed = seed + i
        
        # Get protein info
        protein_info = generator.get_protein_info(gene_name)
        
        # Generate summary
        summary = generator.generate_gene_summary(gene_name, gene_idx, top_n_cells, gene_seed)
        
        result = {
            'gene_name': gene_name,
            'protein_name': protein_info['protein_name'],
            'uniprot_id': protein_info['uniprot_id'],
            'summary': summary
        }
        
        results.append(result)
        
        # Save intermediate results
        if output_dir and i % save_frequency == 0:
            summaries_df = pd.DataFrame(results)
            
            intermediate_pkl = output_dir / f'intermediate_summaries_{i}.pkl'
            with open(intermediate_pkl, 'wb') as f:
                pickle.dump(summaries_df, f)
            
            intermediate_csv = output_dir / f'intermediate_summaries_{i}.csv'
            summaries_df.to_csv(intermediate_csv, index=False)
            
            intermediate_cache = output_dir / f'intermediate_cache_{i}.pkl'
            with open(intermediate_cache, 'wb') as f:
                pickle.dump(generator.protein_cache, f)
            
            print(f"  💾 Saved intermediate results: {i}/{len(hvg_names)} genes")
    
    # Create final DataFrame
    summaries_df = pd.DataFrame(results)
    
    return summaries_df, generator.protein_cache


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate ProCyon-style gene summaries with cellular context for HVGs'
    )
    parser.add_argument('--adata_path', type=str, required=True, help='Path to input AnnData file')
    parser.add_argument('--top_n_cells', type=int, default=100, help='Number of top expressing cells')
    parser.add_argument('--feature_col', type=str, default=None, help='Column for gene names')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tissue', type=str, required=True, help="Tissue name")
    parser.add_argument('--protein_cache_path', type=str, required=False, help="Path to protein cache pickle")
    
    args = parser.parse_args()
    
    # Validate input
    adata_path = Path(args.adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    
    # Create output directory
    data_dir = adata_path.parent
    parent_dir = data_dir.parent
    output_dir = parent_dir / 'summaries' / args.tissue
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("PROCYON GENE SUMMARY GENERATOR")
    print(f"{'='*80}")
    print(f"Input AnnData: {adata_path}")
    print(f"Output directory: {output_dir}")
    print(f"Top N cells per gene: {args.top_n_cells}")
    print(f"{'='*80}\n")
    
    # Load AnnData
    print("Loading AnnData...")
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check for HVGs
    if 'highly_variable' not in adata.var.columns:
        print("Computing highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=4000)
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"Found {n_hvg} highly variable genes")
    
    # Generate summaries
    print("\nGenerating gene summaries...")
    gene_summaries_df, protein_cache = generate_all_gene_summaries_sequential(
        adata,
        top_n_cells=args.top_n_cells,
        feature_col=args.feature_col,
        seed=args.seed,
        output_dir=output_dir,
        protein_cache_path=args.protein_cache_path
    )
    
    # Save outputs
    print("\nSaving final results...")
    summaries_csv_path = output_dir / 'gene_summaries.csv'
    gene_summaries_df.to_csv(summaries_csv_path, index=False)
    print(f"✓ Saved summaries CSV: {summaries_csv_path}")
    
    bundle_path = args.protein_cache_path or output_dir / 'protein_cache.pkl'
    with open(bundle_path, 'wb') as f:
        pickle.dump(protein_cache, f)
    print(f"✓ Saved protein cache: {bundle_path}")
    
    # Clean up intermediate files
    print("\nCleaning up intermediate files...")
    for intermediate_file in output_dir.glob('intermediate_*'):
        intermediate_file.unlink()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Genes processed: {len(gene_summaries_df)}")
    print(f"Proteins mapped: {(gene_summaries_df['uniprot_id'] != '').sum()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()