##
from typing import Dict, Optional
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from scipy.stats import median_abs_deviation
import anndata as ad
import anndata2ri

import json
import os

## Plotting packages --- uncomment for plotting functionalities
# from matplotlib import pyplot as plt
# import seaborn as sns

## import rpy2 packages to run r code
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter

# Suppress rpy2 warnings/messages
rcb.logger.setLevel(logging.ERROR)


sc.settings.verbosity = 0
## The setting below are for plotting, which we do not use in this script for qc 
# sc.settings.set_figure_params(
#     dpi=80,
#     facecolor="white",
#     frameon=False,
# )

# Configure matplotlib globally (affects all plots including scanpy)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


# Configure scanpy to save figures
sc.settings.figdir = '../../figs'  # Directory for saved figures
sc.settings.autosave = True
sc.settings.set_figure_params(
    dpi=300,  # High resolution
    dpi_save=300,  # Save resolution
    facecolor='white',
    figsize=(6, 6),
    transparent=False,
    frameon=False,
    format='pdf'  # Default save format
)



def _run_qc_metrics(adata: ad.AnnData, qc_layer=None, use_raw=False):
    #TODO: add comprehensive documentation
    """
    The first step in quality control is to removes low-quality cells. Low quality cells usaully identified
    using threshold for so-called qc covariate:
    1. The number of counts per barcode (count depth)
    2. The number of genes per barcode
    3. The number of counts from mitochondrial genes per barcode.

    For large datasets, threshold for qc covariates is calculated using median absolute deviation (MAD).
    MAD = median(|X_i - median(X)|, X_i is the respective qc metrics for an observation.
    Cells are counted as outliers if differ by 5 MAD. 
    NB: See the link on top of this script for more information on Quality Control.
    """

    # Get boolean mask for mitochondrial, ribosomal, and hemoglobin  genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.startswith("^HB[^(P)]")

    # calculate the respective qc metrics
    # qc will add new data to the var and obs attributes of adata.
    sc.pp.calculate_qc_metrics(
        adata,
        layer=qc_layer,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[20],
        log1p=False,
        use_raw=use_raw
    )

    dataset_name = adata.uns.get('title', "unamed_adata")

    # QC violin plots
    sc.pl.violin(
    adata, 
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    jitter=0.4, 
    multi_panel=True,
    save=f'{dataset_name}_qc_metrics.pdf'  # Saves as: figures/violin_qc_metrics.pdf
    )

    # calculate outliers
    adata.obs["outlier"] = (
        _is_outlier(adata, "total_counts", 5)
        | _is_outlier(adata, "n_genes_by_counts", 5)
        | _is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    ## Code below can be used to inspect the counts of outliers
    # adata.obs.outlier.value_counts()

    # Filter cells whose mitochondrial genes differ  3 MADs or have percentage mitochondrial counts > 8%
    adata.obs["mt_outlier"] = _is_outlier(adata, "pct_counts_mt", 3) | (adata.obs["pct_counts_mt"] > 8)

    ## can inspect the mt_outlier counts
    # adata.obs.mt_outlier.value_counts()

    # Filter out low qc cell
    ## can print the total cells before filtering
    print(f"Total number of cells: {adata.n_obs}")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    ## can also print the total remaining cells after filtering
    print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")
    ## can also plot a scatter of remaining data
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save=f'{dataset_name}_hqc_scatter.pdf') 


def _is_outlier(adata, metric:str, nmads: int):
    #TODO: Add documentation
    """
    Calculates the Median Absolute Deviation for each qc metrics and mark as outlier
    cells that differ by `nmads`.
    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def _remove_ambient_rna(adata: ad.AnnData, adata_raw: ad.AnnData):
    """
    During scRNA sequencing, some cells RNA without cells, due to lysis, find usually end
    in the same droplet as rna with cells, thus confounding the actual counts of the relevant rna.
    The ambient rnas, usually called soup, can be removed using the soupX package. 

    SoupX is an r package, so we make use of rpy2 to interoperate with r.
    rpy2 helps in running r code in side python.

    Note: We assume that the adata has a layer called 'raw_counts' which contains the raw counts matrix.
    """ 
    adata_pp = adata.copy()
    # soupX requires a bit of clustering
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="soupx_groups", flavor='igraph', directed=False, n_iterations=2)

    soupx_groups = adata_pp.obs["soupx_groups"]
    ## get raw counts from adata. -- assumes that adata has a raw count layer
    data_tod = adata_raw.X.copy().T

    # delete adata_pp and adata_raw to save memory
    del adata_pp
    del adata_raw
    # prepare the data to be passed to r
    pyobj_dict = {
        "cells" : adata.obs_names,
        "genes" : adata.var_names,
        "data_toc" : adata.X.T, # r packages work with transposed matrix.
        "data_tod" : data_tod,
        "soupx_groups" : soupx_groups,

    }

    # convert to r objects
    _pyobjects_to_robject(pyobj_dict)
    

    ## run the r codes now
    r( """
      library(SoupX)
      library(SingleCellExperiment)

      # specify row and column names
      rownames(data_toc) = genes
      colnames(data_toc) = cells

      # ensure correct sparse format for table of counts
      data_toc <- as(data_toc, "sparseMatrix")
      data_tod <- as (data_tod, "sparseMatrix")

      # Generate SoupChannel Object for SoupX
      sc = SoupChannel(data_tod, data_toc, calcSoupProfile = FALSE)

      # Add extra meta data to the soupChannel object
      soupProf = data.frame(row.names = rownames(data_toc), est = rowSums(data_toc)/ sum(data_toc), counts = rowSums(data_toc))
      sc = setSoupProfile(sc, soupProf)
      
      # set Cluster information in soup channel
      sc = setClusters(sc, soupx_groups)

      # Estimate contamination fraction
      sc = autoEstCont(sc, doPlot = FALSE)
      # Infer corrected table of counts and round to integers
      out = adjustCounts(sc, roundToInt = TRUE)
    """)
    # convert data back to python and replace current X with soupx_counts
    adata.layers['filtered_counts'] = adata.X.copy()
    corrected_counts = _robject_to_pyobject('out')
    adata.layers['soupx_counts'] = corrected_counts.T
    adata.X = adata.layers['soupx_counts']

    # Filter out genes not detected in at least 20 cells
    sc.pp.filter_genes(adata, min_cells=20)

def _remove_doublets(adata: ad.AnnData):
    """
    Doublets occur when more than one cells get capture in a gel.
    It is recommended to remove doublets to improve the quality of the gene counts.
    """
    # Get only the counts matrix and convert to r object
    pyobj_dict = {"data_mat" : adata.X.T}
    _pyobjects_to_robject(pyobj_dict)
    r("""
      library(Seurat)
      library(scater)
      library(scDblFinder)
      library(BiocParallel)

      set.seed(123)
      sce = scDblFinder(
        SingleCellExperiment(
            list(counts = data_mat),
        )
      )
      doublet_score = sce$scDblFinder.score
      doublet_class = sce$scDblFinder.class
    """)

    # convert robject back to python object
    adata.obs["scDblFinder_score"] = _robject_to_pyobject("doublet_score")
    adata.obs["scDbleFinder_class"] = _robject_to_pyobject("doublet_class")
    
    ## Can do a value count to inspect the scores
    #adata.obs.scDblFinder_class.value_counts()


### Next we explore three recommend normalization approach
def _log1p(adata:ad.AnnData, layer="X", target_count=1e4):
    """
    Performs log1p normalizaton. Does not renormalized if data is already log1p-normed
    """
    # Save current X values to counts layer
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=None, inplace=True)
    if not _check_logged(adata, layer):
        sc.pp.log1p(adata)
    
    ## can inspect the new distribution by plotting
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
    # axes[0].set_title("Total counts")
    # p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1])
    # axes[1].set_title("Shifted logarithm")
    # plt.show()

def _select_hvgs(adata, flavor='seurat_v3', z_normed=True, batch_key='partition', n_hvg=4000):
    if flavor not in ('seurat_v3', 'cell_ranger'):
        print(f"Flavor not 'seurat_v3' or 'cell_ranger'. Using seurat_v3")
        flavor = 'seurat_v3'
    if flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg, # Select top 4000 genes
            flavor=flavor,
            batch_key='partition',
            subset=False, # Keep all genes, jsut mark HVGs
        )
    
    elif flavor == "cell_ranger":
        sc.pp.highly_variable_genes(
            adata,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            flavor='cell_ranger',
            subset=False
        )
    # Save raw data for future
    adata.raw = adata.copy()
    # Subset to HVGs
    adata = adata[:, adata.var['highly_variable']].copy()

    # Scale subsetted data
    sc.pp.scale(adata, max_value=10)

    # Perform downstream clustering analysis
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # Visualize highly variable genes
    dataset_name = adata.uns.get("title", "unnamed_dataset")
    sc.pl.highly_variable_genes(adata, save=f'{dataset_name}_hvgs.pdf')

    # Scale data to unit variance and zero_mean
    sc.pp.scale(max_value=10)

def _scran_norm(adata: ad.AnnData):
    from scipy.sparse import issparse, csr_matrix
    """
    This normalization has been extensively tested for batch effect correction tasks.
    Scran is implmented in r, so we use the rpy2 interface
    """
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=15)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="groups", flavor="igraph", n_iterations=2, directed=False)

    # add data_mat and computed groups into r environment
    data_mat = adata_pp.X.T
    # convert to CSC if possible. See https://github.com/MarionLab/scran/issues/70
    if issparse(data_mat):
        if data_mat.nnz > 2 ** 31 - 1:
            data_mat = data_mat.tocoo()
        else:
            data_mat = data_mat.tocsc()
    
    pyobjs_dict = {
        "data_mat" : data_mat,
        "input_groups" : adata_pp.obs["groups"],
    }
    _pyobjects_to_robject(pyobjs_dict)

    r("""
        library(scran)
        library(BiocParallel)
        
        size_factors = sizeFactors(
            computeSumFactors(
                SingleCellExperiment(
                    list(counts=data_mat)),
                    clusters = input_groups,
                    min.mean = 0.1,
                    BPPARAM = MulticoreParam(),
            )
        ) 
    """)
    
    # get data back into python object
    adata.obs["size_factors"] = _robject_to_pyobject("size_factors")
    scran = adata.X / adata.obs["size_factors"].values[:, None]
    adata.layers["scran_normalizaton"] = csr_matrix(np.log1p(scran))

    ## Can plot the new distribution
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # p1  = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
    # axes[0].set_title("Total counts")
    # p2 = sns.histplot(
    #     adata.layers["scran_normalization"].sum(1), bins=100, kde=False, ax=axes[1])
    # axes[1].set_title("log1p with Scran estimated size factors")
    # plt.show()

def _analytic_pearson_norm(adata:ad.AnnData):
    from scipy.sparse import csr_matrix
    """"""
    analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(adata, inplace=False)
    adata.layers["analytic_pearson_residuals"] = csr_matrix(analytic_pearson['X'])

    ## Can plot new distributions
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
    # axes[0].set_title("Total counts")
    # p2 = sns.histplot(
    #     adata.layers["analytic_pearson_residuals"].sum(1), bins=100, kde=False, ax=axes[1]
    # )

    # axes[1].set_title("Analytic Pearson residuals")
    # plt.show()


def _select_features(adata:ad.AnnData, layer="X"):
    """
    We used deviance to select feature per the recommendation of sc best practices

    """
    ## We created a new data with just the count matrix because converting the entire adata to robject 
    ## resulted in a memory issue -- apparently anndata2ri is limited to only 16GB data
    adata_copy = ad.AnnData(X=adata.X)
    pyobj = {"adata" : adata_copy}
    _pyobjects_to_robject(pyobj)

    # run the r code
    r("""
        library(scry)
        sce = devianceFeatureSelection(adata, assay="X")
        binomial_deviance = rowData(sce)$binomial_deviance
    """)

    bin_dev = _robject_to_pyobject("binomial_deviance")
    binomial_deviance = bin_dev.T

    # select the 4000 top highly deviant genes
    idx = binomial_deviance.argsort()[-4000:]
    mask = np.zeros(adata.var_names.shape, dtype=bool)
    mask[idx] = True

    adata.var["highly_deviant"] = mask
    adata.var["binomial_deviance"] = binomial_deviance

    sc.pp.highly_variable_genes(adata, layer=layer)

    ## Can visualize the feature selection result
    # ax = sns.scatterplot(
    # data=adata.var, x="means", y="dispersions", hue="highly_deviant", s=5
    # )

def run_qc_pipeline(adata: ad.AnnData, layer="X", use_raw=False,):
    """
    Run the entire qc pipeline.
    We only perform one of the three normalization methods: scran normalization.
    1. Run qc metrics and filter low quality cells
    2. Remove ambient rna using soupX
    3. Remove doublets using scDblFinder
    4. Normalize using analytic pearson residuals for the selection of highly variable genes
    5. Select highly variable genes using deviance feature selection 
    """
    # _run_qc_metrics(adata)
    # _remove_ambient_rna(adata, adata_raw)
    # _remove_doublets(adata)
    # _scran_norm(adata)
    # _analytic_pearson_norm(adata)
    # _select_features(adata, layer="analytic_pearson_residuals")

    _run_qc_metrics(adata)
    _log1p(adata)
    _select_hvgs(adata)



def _check_logged(self, adata: ad.AnnData, layer: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        layer (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.

        Credits:
            This method is taken from scgpt preprocessing.py
        """
        layer = layer or "X"
        data = adata.layers[layer]
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True

def _pyobjects_to_robject(adatas: Dict[str, ad.AnnData]):
    with localconverter(anndata2ri.converter):
        for key, val in adatas.items():
            ro.globalenv[key] = val

def _pyobject_to_robject(adata):
    with localconverter(anndata2ri.converter):
        ro.globalenv['sce'] = adata

def _robject_to_pyobject(name: str):
    with localconverter(anndata2ri.converter):
        return ro.globalenv[name]
    

# Run the main module
if __name__ == "__main__":
    # load configuration  json file
    config = json.load(open("../datapaths.json", "r"))
    datasets_dirpath = config['datasets_dir']
    outputs_dirpath = config['outputs_dir']
    mat_ouput_dirpath = os.path.join(outputs_dirpath, 'matrices')
    adata_output_dirpath = os.path.join(outputs_dirpath, 'adatas')

    # create the directories
    if not os.path.exists(outputs_dirpath):
        os.makedirs(outputs_dirpath)
    if not os.path.exists(adata_output_dirpath):
        os.makedirs(adata_output_dirpath)
    

    # Do some sanity check on the config file
    for fp in config['files']:
         # load the filtered data
        adata = sc.read_10x_mtx(
            path=os.path.join(datasets_dirpath, fp['filteredpath']),
            var_names="gene_symbols",
            make_unique=True
        )
        # make var names unique
        #adata.var_names_make_unique()
        
        # load the raw data
        adata_raw = sc.read_10x_mtx(
        path=os.path.join(datasets_dirpath, fp['rawpath']),
        var_names="gene_symbols",
        make_unique=True

        )
        #adata_raw.var_names_make_unique()
        
        # Run QC metrics
        run_qc_pipeline(adata, adata_raw)
        # Save the processed data
        adata.write_h5ad(os.path.join(adata_output_dirpath, fp['name'] + '.h5ad'))
        # Save only the feature matrices
        adata.to_df().to_csv(os.path.join(mat_ouput_dirpath, fp['name']+'.csv'))

    

