import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import os
from pathlib import Path
import scanpy as sc

class CellSummaryGenerator:
    """Generate rich, ProCyon-style instructional summaries for cells with sentence variation."""
    
    # Sentence templates for variation
    INTRO_TEMPLATES = [
        "This cell population consists of {n_cells:,} {cell_label} cells. ",
        "The dataset contains {n_cells:,} {cell_label} cells. ",
        "This analysis identifies {n_cells:,} {cell_label} cells within the sample. ",
        "A total of {n_cells:,} {cell_label} cells were characterized. ",
        "The cell population comprises {n_cells:,} {cell_label} cells. "
    ]
    
    EXPRESSION_TEMPLATES = [
        "These cells exhibit an average expression profile with {mean_counts:,.0f} ± {std_counts:,.0f} total transcript counts and express approximately {mean_genes:.0f} distinct genes per cell. ",
        "Transcriptomic profiling reveals {mean_counts:,.0f} ± {std_counts:,.0f} total counts per cell, with approximately {mean_genes:.0f} genes actively expressed. ",
        "Expression analysis shows an average of {mean_counts:,.0f} ± {std_counts:,.0f} transcript counts across {mean_genes:.0f} detected genes per cell. ",
        "The cells display {mean_counts:,.0f} ± {std_counts:,.0f} total transcript counts on average, expressing roughly {mean_genes:.0f} unique genes. ",
        "Molecular profiling indicates {mean_counts:,.0f} ± {std_counts:,.0f} total transcripts per cell, with {mean_genes:.0f} genes showing detectable expression. "
    ]
    
    MARKER_TEMPLATES = [
        "The most distinctive marker genes for this cell type include {markers}, which are specifically enriched compared to other cell populations. ",
        "Key marker genes such as {markers} show specific enrichment in this cell type relative to others. ",
        "Differential expression analysis highlights {markers} as characteristic markers for this population. ",
        "This cell type is distinguished by elevated expression of {markers} compared to other populations. ",
        "Cell type-specific markers include {markers}, which display preferential expression in these cells. "
    ]
    
    TOP_GENES_TEMPLATES = [
        "The most highly expressed genes include {genes}, reflecting the cell type's functional specialization and metabolic activity. ",
        "Abundant transcripts comprise {genes}, indicative of the cell's functional state and metabolic demands. ",
        "Top expressed genes such as {genes} underscore the cell type's specialized biological functions. ",
        "High expression levels of {genes} characterize the transcriptional program of these cells. ",
        "The transcriptome is dominated by {genes}, revealing key aspects of cellular function and identity. "
    ]
    
    PROTEIN_INTRO_TEMPLATES = [
        "At the protein level, these cells display characteristic molecular signatures. ",
        "Protein-level analysis reveals distinctive molecular features. ",
        "The proteomic landscape of these cells shows defining characteristics. ",
        "Key protein components define the molecular identity of this cell type. ",
        "Protein expression patterns provide molecular insights into cell function. "
    ]
    
    def __init__(self, adata, protein_cache=None):
        self.adata = adata
        self.gene_names = adata.var['feature_name'].values
        self.protein_cache = protein_cache if protein_cache else {}
        
    def get_protein_info(self, gene_name: str) -> Optional[Dict]:
        """Retrieve protein information from UniProt."""
        if gene_name in self.protein_cache:
            return self.protein_cache[gene_name]
        
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': f'gene:{gene_name} AND organism_name:human',
            'format': 'json',
            'fields': 'accession,gene_names,protein_name,cc_function,cc_subcellular_location',
            'size': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.ok:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    
                    protein_name = ''
                    if 'proteinDescription' in result:
                        rec_name = result['proteinDescription'].get('recommendedName', {})
                        protein_name = rec_name.get('fullName', {}).get('value', '')
                    
                    function = ''
                    if 'comments' in result:
                        for comment in result['comments']:
                            if comment.get('commentType') == 'FUNCTION':
                                texts = comment.get('texts', [])
                                if texts:
                                    function = texts[0].get('value', '')
                                break
                    
                    location = ''
                    if 'comments' in result:
                        for comment in result['comments']:
                            if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                                subcell_locs = comment.get('subcellularLocations', [])
                                if subcell_locs:
                                    loc_obj = subcell_locs[0].get('location', {})
                                    location = loc_obj.get('value', '')
                                break
                    
                    info = {
                        'gene': gene_name,
                        'protein_name': protein_name,
                        'function': function,
                        'location': location,
                        'uniprot_id': result.get('primaryAccession', '')
                    }
                    
                    self.protein_cache[gene_name] = info
                    return info
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching protein info for {gene_name}: {e}")
        
        return None
    
    def get_batch_protein_info(self, gene_list: List[str]) -> Dict[str, Dict]:
        """Get protein information for multiple genes efficiently."""
        protein_info = {}
        
        for gene in gene_list:
            info = self.get_protein_info(gene)
            if info:
                protein_info[gene] = info
        
        return protein_info
    
    def get_cell_ontology_info(self, cell_type: str) -> Dict:
        """Retrieve detailed cell type information from Cell Ontology."""
        url = "https://www.ebi.ac.uk/ols/api/search"
        params = {
            'q': cell_type,
            'ontology': 'cl',
            'exact': 'false',
            'rows': 1
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.ok:
                data = response.json()
                if data['response']['docs']:
                    doc = data['response']['docs'][0]
                    return {
                        'label': doc.get('label', cell_type),
                        'definition': doc.get('description', [''])[0] if doc.get('description') else '',
                        'synonyms': doc.get('synonym', []),
                        'cl_id': doc.get('obo_id', '')
                    }
        except Exception as e:
            print(f"Error fetching ontology for {cell_type}: {e}")
        
        return {'label': cell_type, 'definition': '', 'synonyms': [], 'cl_id': ''}
    
    def get_top_expressed_genes(self, cell_mask: np.ndarray, top_n: int = 20) -> List[tuple]:
        """Get top expressed genes for a cell subset."""
        cell_subset = self.adata[cell_mask]
        
        if hasattr(cell_subset.X, 'toarray'):
            mean_expr = np.array(cell_subset.X.mean(axis=0)).flatten()
        else:
            mean_expr = cell_subset.X.mean(axis=0)
        
        top_indices = np.argsort(mean_expr)[-top_n:][::-1]
        top_genes = [(self.gene_names[i], mean_expr[i]) for i in top_indices]
        
        return top_genes
    
    def get_differential_markers(self, cell_mask: np.ndarray, top_n: int = 10) -> List[str]:
        """Get genes specifically enriched in this cell type."""
        in_group = self.adata[cell_mask].X.mean(axis=0)
        out_group = self.adata[~cell_mask].X.mean(axis=0)
        
        if hasattr(in_group, 'A1'):
            in_group = in_group.A1
            out_group = out_group.A1
        
        fold_change = np.log2((in_group + 1) / (out_group + 1))
        top_markers = np.argsort(fold_change)[-top_n:][::-1]
        
        return [self.gene_names[i] for i in top_markers]
    
    def generate_procyon_style_summary(self, cell_type: str, cell_mask: np.ndarray, 
                                      seed: Optional[int] = None) -> str:
        """
        Generate a rich, instructional summary with randomized sentence templates.
        
        Args:
            cell_type: Name of the cell type
            cell_mask: Boolean mask for selecting cells
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
        
        # Get ontology information
        ontology_info = self.get_cell_ontology_info(cell_type)
        
        # Get expression data
        top_genes = self.get_top_expressed_genes(cell_mask, top_n=20)
        marker_genes = self.get_differential_markers(cell_mask, top_n=10)
        
        # Get protein information for top 10 genes
        top_10_genes = [gene for gene, _ in top_genes[:10]]
        protein_info = self.get_batch_protein_info(top_10_genes)
        
        # Calculate statistics
        n_cells = cell_mask.sum()
        cell_subset = self.adata[cell_mask]
        
        if 'total_counts' in cell_subset.obs.columns:
            mean_counts = cell_subset.obs['total_counts'].mean()
            std_counts = cell_subset.obs['total_counts'].std()
        else:
            mean_counts = cell_subset.X.sum(axis=1).mean()
            std_counts = cell_subset.X.sum(axis=1).std()
        
        if 'n_genes_by_counts' in cell_subset.obs.columns:
            mean_genes = cell_subset.obs['n_genes_by_counts'].mean()
        else:
            mean_genes = (cell_subset.X > 0).sum(axis=1).mean()
        
        # Build the summary with randomized templates
        summary_parts = []
        
        # 1. Cell type introduction (randomized)
        cell_label = ontology_info['label']
        intro = random.choice(self.INTRO_TEMPLATES).format(
            n_cells=n_cells, 
            cell_label=cell_label
        )
        summary_parts.append(intro)
        
        # 2. Biological definition (always include if available)
        if ontology_info['definition']:
            summary_parts.append(f"{ontology_info['definition']} ")
        
        # 3. Expression characteristics (randomized)
        expr = random.choice(self.EXPRESSION_TEMPLATES).format(
            mean_counts=mean_counts,
            std_counts=std_counts,
            mean_genes=mean_genes
        )
        summary_parts.append(expr)
        
        # 4. Key marker genes (randomized)
        top_5_markers = marker_genes[:5]
        if len(top_5_markers) > 0:
            marker_str = ", ".join(top_5_markers[:-1]) + f", and {top_5_markers[-1]}"
            marker = random.choice(self.MARKER_TEMPLATES).format(markers=marker_str)
            summary_parts.append(marker)
        
        # 5. Highly expressed genes (randomized)
        top_5_expressed = [gene for gene, _ in top_genes[:5]]
        if len(top_5_expressed) > 0:
            genes_str = ", ".join(top_5_expressed[:-1]) + f", and {top_5_expressed[-1]}"
            genes = random.choice(self.TOP_GENES_TEMPLATES).format(genes=genes_str)
            summary_parts.append(genes)
        
        # 6. Protein information (randomized intro)
        if protein_info:
            protein_intro = random.choice(self.PROTEIN_INTRO_TEMPLATES)
            summary_parts.append(protein_intro)
            
            protein_descriptions = []
            for gene, info in list(protein_info.items())[:5]:
                if info['protein_name'] and info['function']:
                    func = info['function'].split('.')[0] + '.'
                    desc = f"{info['protein_name']} ({gene}), {func}"
                    if info['location']:
                        desc += f" This protein is localized to the {info['location'].lower()}."
                    protein_descriptions.append(desc)
                elif info['protein_name']:
                    protein_descriptions.append(f"{info['protein_name']} ({gene})")
            
            if protein_descriptions:
                summary_parts.append("Key proteins include: " + " ".join(protein_descriptions) + " ")
        
        # 7. Alternative nomenclature (randomly include 50% of the time)
        if ontology_info['synonyms'] and random.random() > 0.5:
            syn_str = ", ".join(ontology_info['synonyms'][:3])
            summary_parts.append(
                f"This cell type may also be referred to as {syn_str} in the literature. "
            )
        
        # 8. Ontology reference (always include if available)
        if ontology_info['cl_id']:
            summary_parts.append(f"Cell Ontology ID: {ontology_info['cl_id']}.")
        
        return "".join(summary_parts)


def process_cell_type(args: Tuple) -> Dict:
    """
    Worker function for parallel processing.
    
    Args:
        args: Tuple of (cell_type, cell_mask_indices, adata_subset, protein_cache, seed)
    
    Returns:
        Dictionary with cell_type, n_cells, summary, and protein IDs
    """
    cell_type, cell_mask_indices, X_data, obs_data, var_data, protein_cache, seed = args
    
    # Reconstruct a minimal AnnData-like structure
    class MinimalAdata:
        def __init__(self, X, obs, var):
            self.X = X
            self.obs = obs
            self.var = var
            self.var['feature_name'] = var['feature_name']
    
    mini_adata = MinimalAdata(X_data, obs_data, var_data)
    
    # Create generator with shared protein cache
    generator = CellSummaryGenerator(mini_adata, protein_cache=protein_cache)
    
    # Create cell mask
    cell_mask = np.zeros(len(obs_data), dtype=bool)
    cell_mask[cell_mask_indices] = True
    
    print(f"Processing {cell_type}...")
    
    # Get top genes for protein tracking
    top_genes = generator.get_top_expressed_genes(cell_mask, top_n=20)
    top_10_genes = [gene for gene, _ in top_genes[:10]]
    
    # Get protein info
    protein_info = generator.get_batch_protein_info(top_10_genes)
    
    # Extract UniProt IDs
    uniprot_ids = []
    gene_to_uniprot = {}
    for gene, info in protein_info.items():
        if info and info.get('uniprot_id'):
            uniprot_ids.append(info['uniprot_id'])
            gene_to_uniprot[gene] = info['uniprot_id']
    
    # Generate summary
    summary = generator.generate_procyon_style_summary(cell_type, cell_mask, seed=seed)
    
    return {
        'cell_type': cell_type,
        'n_cells': cell_mask.sum(),
        'summary': summary,
        'uniprot_ids': uniprot_ids,  # List of UniProt IDs
        'gene_to_uniprot': gene_to_uniprot,  # Gene name -> UniProt ID mapping
        'top_genes': top_10_genes,  # Top 10 gene names
        'protein_cache_update': generator.protein_cache
    }


def generate_all_summaries_parallel(adata, cell_type_col: str = 'cell_type', 
                                   n_processes: Optional[int] = None,
                                   seed: int = 42) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Generate ProCyon-style summaries for all cell types in parallel.
    
    Args:
        adata: AnnData object
        cell_type_col: Column name containing cell type annotations
        n_processes: Number of parallel processes (defaults to CPU count)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - DataFrame with cell type summaries
        - Protein cache dictionary
        - Cell type to UniProt IDs mapping
    """
    if cell_type_col not in adata.obs.columns:
        possible_cols = [col for col in adata.obs.columns 
                       if 'cell' in col.lower() and 'type' in col.lower()]
        if possible_cols:
            cell_type_col = possible_cols[0]
            print(f"Using column: {cell_type_col}")
        else:
            raise ValueError("No cell type column found!")
    
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"Using {n_processes} processes for parallel processing...")
    
    unique_cell_types = adata.obs[cell_type_col].unique()
    
    # Prepare arguments for parallel processing
    args_list = []
    shared_protein_cache = {}
    
    for i, cell_type in enumerate(unique_cell_types):
        cell_mask_indices = np.where(adata.obs[cell_type_col] == cell_type)[0]
        
        # Use different seeds for each cell type for variation
        cell_seed = seed + i
        
        args_list.append((
            cell_type,
            cell_mask_indices,
            adata.X,
            adata.obs,
            adata.var,
            shared_protein_cache,
            cell_seed
        ))
    
    # Process in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_cell_type, args_list)
    
    # Merge protein caches
    final_protein_cache = {}
    for result in results:
        final_protein_cache.update(result['protein_cache_update'])
    
    # Create cell type to UniProt mapping
    cell_type_to_uniprots = {}
    for result in results:
        cell_type_to_uniprots[result['cell_type']] = {
            'uniprot_ids': result['uniprot_ids'],
            'gene_to_uniprot': result['gene_to_uniprot'],
            'top_genes': result['top_genes']
        }
    
    # Create DataFrame
    summaries_df = pd.DataFrame([
        {
            'cell_type': r['cell_type'], 
            'n_cells': r['n_cells'], 
            'summary': r['summary'],
            'uniprot_ids': ','.join(r['uniprot_ids']),  # Comma-separated string
            'n_proteins': len(r['uniprot_ids']),
            'top_genes': ','.join(r['top_genes'])
        }
        for r in results
    ])
    
    return summaries_df, final_protein_cache, cell_type_to_uniprots


# Usage Example
# ==============

def create_summary_adata(adata, cell_summaries_df, cell_type_to_uniprots, 
                        uniprot_matrix, uniprot_id_list):
    """
    Create a new AnnData object with cell embeddings and summary information.
    
    Args:
        adata: Original AnnData object
        cell_summaries_df: DataFrame with cell type summaries
        cell_type_to_uniprots: Dictionary mapping cell types to UniProt info
        uniprot_matrix: Binary matrix of UniProt IDs per cell
        uniprot_id_list: List of UniProt ID names
    
    Returns:
        New AnnData object with processed data
    """
    # Create new AnnData with expression matrix
    adata_summary = sc.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    
    # Add cell summaries
    cell_type_to_summary = dict(zip(cell_summaries_df['cell_type'], 
                                    cell_summaries_df['summary']))
    adata_summary.obs['cell_summary'] = adata_summary.obs['cell_type'].map(cell_type_to_summary)
    
    # Add UniProt IDs as comma-separated string
    cell_type_to_uniprot_str = dict(zip(cell_summaries_df['cell_type'],
                                        cell_summaries_df['uniprot_ids']))
    adata_summary.obs['uniprot_ids'] = adata_summary.obs['cell_type'].map(cell_type_to_uniprot_str)
    
    # Add number of proteins per cell type
    cell_type_to_n_proteins = dict(zip(cell_summaries_df['cell_type'],
                                       cell_summaries_df['n_proteins']))
    adata_summary.obs['n_proteins'] = adata_summary.obs['cell_type'].map(cell_type_to_n_proteins)
    
    # Add top genes per cell type
    cell_type_to_top_genes = dict(zip(cell_summaries_df['cell_type'],
                                      cell_summaries_df['top_genes']))
    adata_summary.obs['top_genes'] = adata_summary.obs['cell_type'].map(cell_type_to_top_genes)
    
    # Add UniProt binary matrix to obsm
    adata_summary.obsm['uniprot_ids_binary'] = uniprot_matrix
    
    # Store metadata in uns
    adata_summary.uns['cell_type_summaries'] = cell_summaries_df.to_dict('records')
    adata_summary.uns['cell_type_to_uniprots'] = cell_type_to_uniprots
    adata_summary.uns['uniprot_id_names'] = uniprot_id_list
    
    return adata_summary


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate ProCyon-style cell summaries with protein information'
    )
    parser.add_argument(
        '--adata_path',
        type=str,
        required=True,
        help='Path to input AnnData file (.h5ad)'
    )
    parser.add_argument(
        '--cell_type_col',
        type=str,
        default='cell_type',
        help='Column name containing cell type annotations (default: cell_type)'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: CPU count - 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    adata_path = Path(args.adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    
    # Create output directory: parent_dir/summaries/
    data_dir = adata_path.parent
    parent_dir = data_dir.parent
    output_dir = parent_dir / 'summaries'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("PROCYON CELL SUMMARY GENERATOR")
    print(f"{'='*80}")
    print(f"Input AnnData: {adata_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cell type column: {args.cell_type_col}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Load AnnData
    print("Loading AnnData...")
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Generate summaries in parallel with UniProt ID tracking
    print("\nGenerating cell type summaries...")
    cell_summaries_df, protein_cache, cell_type_to_uniprots = generate_all_summaries_parallel(
        adata, 
        cell_type_col=args.cell_type_col,
        n_processes=args.n_processes,
        seed=args.seed
    )
    
    # Create UniProt ID matrix
    print("\nCreating UniProt ID matrix for ProCyon...")
    all_uniprot_ids = set()
    for info in cell_type_to_uniprots.values():
        all_uniprot_ids.update(info['uniprot_ids'])
    
    uniprot_id_list = sorted(list(all_uniprot_ids))
    uniprot_to_idx = {uid: i for i, uid in enumerate(uniprot_id_list)}
    
    # Create binary matrix: cells x UniProt IDs
    uniprot_matrix = np.zeros((adata.n_obs, len(uniprot_id_list)), dtype=np.int8)
    
    for i, cell_type in enumerate(adata.obs[args.cell_type_col]):
        cell_uniprots = cell_type_to_uniprots.get(cell_type, {}).get('uniprot_ids', [])
        for uid in cell_uniprots:
            if uid in uniprot_to_idx:
                uniprot_matrix[i, uniprot_to_idx[uid]] = 1
    
    print(f"Created UniProt ID matrix: {uniprot_matrix.shape}")
    print(f"Total unique UniProt IDs: {len(uniprot_id_list)}")
    print(f"Average proteins per cell: {uniprot_matrix.sum(axis=1).mean():.1f}")
    
    # Create new AnnData with summaries
    print("\nCreating summary AnnData object...")
    adata_summary = create_summary_adata(
        adata, 
        cell_summaries_df, 
        cell_type_to_uniprots,
        uniprot_matrix,
        uniprot_id_list
    )
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Save summary AnnData
    summary_adata_path = output_dir / f'{adata_path.stem}_with_summaries.h5ad'
    adata_summary.write_h5ad(summary_adata_path)
    print(f"✓ Saved summary AnnData: {summary_adata_path}")
    
    # Save summaries CSV
    summaries_csv_path = output_dir / 'cell_type_summaries.csv'
    cell_summaries_df.to_csv(summaries_csv_path, index=False)
    print(f"✓ Saved summaries CSV: {summaries_csv_path}")
    
    # Save protein cache
    import pickle
    protein_cache_path = output_dir / 'protein_info_cache.pkl'
    with open(protein_cache_path, 'wb') as f:
        pickle.dump(protein_cache, f)
    print(f"✓ Saved protein cache: {protein_cache_path}")
    
    # Save cell type to UniProt mapping
    uniprot_mapping_path = output_dir / 'cell_type_to_uniprots.pkl'
    with open(uniprot_mapping_path, 'wb') as f:
        pickle.dump(cell_type_to_uniprots, f)
    print(f"✓ Saved UniProt mapping: {uniprot_mapping_path}")
    
    # Save UniProt ID list
    uniprot_list_path = output_dir / 'uniprot_id_list.txt'
    with open(uniprot_list_path, 'w') as f:
        f.write('\n'.join(uniprot_id_list))
    print(f"✓ Saved UniProt ID list: {uniprot_list_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Input: {adata_path}")
    print(f"Output directory: {output_dir}")
    print(f"")
    print(f"Cell types processed: {len(cell_summaries_df)}")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Total genes: {adata.n_vars:,}")
    print(f"Unique proteins tracked: {len(protein_cache)}")
    print(f"UniProt IDs mapped: {len(uniprot_id_list)}")
    print(f"")
    print("Output files:")
    print(f"  - {summary_adata_path.name}")
    print(f"  - {summaries_csv_path.name}")
    print(f"  - {protein_cache_path.name}")
    print(f"  - {uniprot_mapping_path.name}")
    print(f"  - {uniprot_list_path.name}")
    print(f"{'='*80}\n")
    
    # Show example summaries
    print("Example cell type summaries:\n")
    for idx, row in cell_summaries_df.head(3).iterrows():
        print(f"{'='*80}")
        print(f"Cell Type: {row['cell_type']}")
        print(f"Cells: {row['n_cells']:,} | Proteins: {row['n_proteins']}")
        print(f"{'='*80}")
        print(row['summary'][:500] + "..." if len(row['summary']) > 500 else row['summary'])
        print()


if __name__ == "__main__":
    main()