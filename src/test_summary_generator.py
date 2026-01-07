import scanpy as sc
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd

def test_gene_summary_generator(adata_path: str, 
                                n_test_cells: int = 10,
                                n_test_hvgs: int = 50,
                                top_n_cells: int = 10,
                                feature_col="gene_ids",
                                output_dir: Optional[str] = None,
                                seed: int = 42):
    """
    Test the GeneSummaryGenerator on a small subset of data.
    
    Args:
        adata_path: Path to full AnnData file
        n_test_cells: Number of cells to test with (default: 10)
        n_test_hvgs: Number of HVGs to test with (default: 50)
        top_n_cells: Number of top cells per gene (default: 10)
        output_dir: Output directory (default: './trial_summaries')
        seed: Random seed
    """
    from generate_gene_summaries import (
        GeneSummaryGenerator, 
        generate_all_gene_summaries_parallel,
        create_summary_adata
    )
    
    print("="*80)
    print("GENE SUMMARY GENERATOR - TRIAL RUN")
    print("="*80)
    
    # Setup
    np.random.seed(seed)
    if output_dir is None:
        output_dir = Path('./trial_summaries')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"\n1. Loading AnnData from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"   Original data: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Create subset
    print(f"\n2. Creating trial subset: {n_test_cells} cells")
    cell_indices = np.random.choice(adata.n_obs, size=min(n_test_cells, adata.n_obs), replace=False)
    adata_trial = adata[cell_indices].copy()
    
    # Setup HVGs
    print(f"\n3. Setting up {n_test_hvgs} highly variable genes")
    if 'highly_variable' not in adata_trial.var.columns:
        sc.pp.highly_variable_genes(adata_trial, n_top_genes=n_test_hvgs)
    else:
        # Keep only top N HVGs
        hvg_genes = adata_trial.var[adata_trial.var['highly_variable']].copy()
        if len(hvg_genes) > n_test_hvgs:
            if 'variances_norm' in hvg_genes.columns:
                top_hvg = hvg_genes.nlargest(n_test_hvgs, 'variances_norm').index
            else:
                top_hvg = hvg_genes.index[:n_test_hvgs]
            
            adata_trial.var['highly_variable'] = False
            adata_trial.var.loc[top_hvg, 'highly_variable'] = True
    
    n_hvg = adata_trial.var['highly_variable'].sum()
    hvg_names = adata_trial.var['gene_ids'][adata_trial.var['highly_variable']].tolist()
    print(f"   HVGs selected: {n_hvg}")
    print(f"   Example HVGs: {hvg_names[:5]}")
    
    # Test GeneSummaryGenerator directly
    print(f"\n4. Testing GeneSummaryGenerator class")
    print("-"*80)
    
    try:
        generator = GeneSummaryGenerator(adata_trial, output_dir=output_dir)
        print(f"   ✓ Generator initialized")
        print(f"   ✓ Found {len(generator.hvg_names)} HVGs")
        print(f"   ✓ Cell type column: {generator.cell_type_col}")
        
        # Test on first HVG
        test_gene_name = generator.hvg_names[0]
        test_gene_idx = generator.hvg_indices[0]
        
        print(f"\n   Testing single gene: {test_gene_name}")
        
        # Test protein info retrieval
        protein_info = generator.get_protein_info(test_gene_name)
        print(f"   ✓ Protein info retrieved")
        print(f"      - Protein name: {protein_info['protein_name']}")
        print(f"      - UniProt ID: {protein_info['uniprot_id']}")
        print(f"      - Function: {protein_info['function'][:100]}..." if protein_info['function'] else "      - Function: N/A")
        
        # Test top cells
        top_cell_indices = generator.get_top_expressing_cells(test_gene_idx, top_n_cells)
        print(f"   ✓ Top cells identified: {len(top_cell_indices)}")
        
        # Test co-expression
        coexpr_genes = generator.get_coexpressed_genes(test_gene_idx, top_cell_indices, top_n=5)
        print(f"   ✓ Co-expressed genes: {coexpr_genes}")
        
        # Test cell types
        top_cell_types = generator.get_top_cell_types(top_cell_indices, top_n=3)
        print(f"   ✓ Top cell types: {top_cell_types if top_cell_types else 'N/A'}")
        
        # Generate summary
        summary = generator.generate_gene_summary(test_gene_name, test_gene_idx, top_n_cells, seed)
        print(f"   ✓ Summary generated ({len(summary)} chars)")
        
        print(f"\n   Sample Summary:")
        print(f"   {'-'*76}")
        print(f"   {summary[:300]}...")
        print(f"   {'-'*76}")
        
    except Exception as e:
        print(f"   ✗ Error in generator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test parallel processing
    print(f"\n5. Testing parallel processing on all {n_hvg} HVGs")
    print("-"*80)
    
    try:
        gene_summaries_df, protein_cache = generate_all_gene_summaries_parallel(
            adata_trial,
            n_processes=40,
            top_n_cells=top_n_cells,
            seed=seed,
            feature_col=feature_col,
            output_dir=output_dir
        )
        
        print(f"   ✓ Parallel processing complete")
        print(f"   ✓ Summaries generated: {len(gene_summaries_df)}")
        print(f"   ✓ Proteins cached: {len(protein_cache)}")
        print(f"   ✓ Proteins with UniProt IDs: {(gene_summaries_df['uniprot_id'] != '').sum()}")
        
    except Exception as e:
        print(f"   ✗ Error in parallel processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create summary AnnData
    print(f"\n6. Creating summary AnnData")
    print("-"*80)
    
    # try:
    #     # adata_summary = create_summary_adata(adata_trial, gene_summaries_df, protein_cache, feature_col)
    #     # print(f"   ✓ Summary AnnData created")
    #     # print(f"   ✓ Shape: {adata_summary.shape}")
    #     # print(f"   ✓ New var columns: {[c for c in adata_summary.var.columns if 'summary' in c.lower() or 'uniprot' in c.lower() or 'protein' in c.lower()]}")
        
    # except Exception as e:
    #     print(f"   ✗ Error creating summary AnnData: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return
    
    # Save outputs
    print(f"\n7. Saving outputs to: {output_dir}")
    print("-"*80)
    
    # Save pickle bundle
    bundle_path = output_dir / 'trial_gene_summaries_bundle.pkl'
    bundle = {
        'summaries_df': gene_summaries_df,
        'protein_cache': protein_cache,
        'hvg_names': hvg_names,
        'top_n_cells': top_n_cells,
        'n_cells': adata_trial.n_obs,
        'n_genes': adata_trial.n_vars,
        'n_hvg': n_hvg
    }
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"   ✓ Saved bundle: {bundle_path}")
    
    # Save DataFrame
    df_path = output_dir / 'trial_gene_summaries.pkl'
    with open(df_path, 'wb') as f:
        pickle.dump(gene_summaries_df, f)
    print(f"   ✓ Saved DataFrame: {df_path}")
    
    # Save CSV
    csv_path = output_dir / 'trial_gene_summaries.csv'
    gene_summaries_df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved CSV: {csv_path}")
    
    # Save AnnData
    # adata_path = output_dir / 'trial_adata_with_summaries.h5ad'
    # adata_summary.write_h5ad(adata_path)
    # print(f"   ✓ Saved AnnData: {adata_path}")
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nDataFrame columns:")
    print(f"  {list(gene_summaries_df.columns)}")
    
    print(f"\nStatistics:")
    # print(f"  - Genes processed: {len(gene_summaries_df)}")
    # print(f"  - Avg expressing cells: {gene_summaries_df['n_expressing_cells'].mean():.1f}")
    # print(f"  - Avg expression: {gene_summaries_df['mean_expression'].mean():.2f}")
    # print(f"  - Proteins with UniProt: {(gene_summaries_df['uniprot_id'] != '').sum()}")
    
    print(f"\nSample summaries (first 3 genes):")
    print("="*80)
    
    # for idx, row in gene_summaries_df.head(3).iterrows():
    #     print(f"\nGene: {row['gene_name']}")
    #     print(f"Protein: {row['protein_name']}")
    #     print(f"UniProt: {row['uniprot_id']}")
    #     print(f"Expressing cells: {row['n_expressing_cells']}")
    #     print(f"Top cell types: {row['top_cell_types']}")
    #     print(f"{'-'*80}")
    #     summary_text = row['summary']
    #     print(summary_text[:400] + "..." if len(summary_text) > 400 else summary_text)
    
    # print(f"\n{'='*80}")
    # print("✓ TRIAL COMPLETE - ALL TESTS PASSED!")
    # print(f"{'='*80}")
    # print(f"\nTo run on full dataset:")
    # print(f"python generate_gene_summaries.py \\")
    # print(f"    --adata_path {adata_path} \\")
    # print(f"    --top_n_cells 22000 \\")
    # print(f"    --n_processes 8")
    
    return gene_summaries_df, protein_cache


# Usage
if __name__ == "__main__":
    # Test the generator
    adata_path = "/users/madorsoo/scratch/procupine/preprocessed/adatas/5k_pbmc_protein.h5ad"
    
    summaries_df, protein_cache, = test_gene_summary_generator(
        adata_path=adata_path,
        n_test_cells=10,
        n_test_hvgs=10,
        top_n_cells=10,
        output_dir="./trial_summaries",
        feature_col="gene_ids",
        seed=42
    )
    
    # Quick inspection
    print("\n" + "="*80)
    print("Quick inspection of results:")
    print("="*80)
    print(f"Summaries DataFrame shape: {summaries_df.shape}")
    print(f"Protein cache size: {len(protein_cache)}")
    # print(f"AnnData shape: {adata_summary.shape}")