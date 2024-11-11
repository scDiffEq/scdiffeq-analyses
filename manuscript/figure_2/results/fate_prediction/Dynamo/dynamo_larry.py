#!/usr/bin/python
from argparse import ArgumentParser

import numpy as np
import dynamo as dyn
import anndata


import os
os.chdir("/home/ruitong/scDiffEq_PyTorch/finaltouch")

import sys
sys.path.insert(0, "/home/ruitong/scDiffEq_PyTorch/finaltouch/")
from fxns.utils import *

def main(hparams):
    adata_file = "data/train_larry_dyn.h5ad"
    try:
        train_dyn = anndata.read_h5ad(adata_file)
    except:
        larryann = anndata.read_h5ad("data/larry_full.h5ad")
        CBidx = larryann.obs.Library.astype('str') + ":" + \
            larryann.obs["Cell barcode"].astype("str")
        CBidx = CBidx.str.replace("\\-", '', regex=True).values
        larryann.obs.index = CBidx
        # load the larry dataset with spliced/unspliced reads
        from pyrovelocity import data
        pv_larry = data.load_larry( ) 
        pv_larry.obs = larryann[pv_larry.obs.index].obs
        train_dyn = pv_larry[pv_larry.obs.Well!=2]
        train_dyn.write_h5ad("data/train_larry_dyn.h5ad")
    
    s = int(hparams.seed)
    np.random.seed(s)
    ## pre-proc
    dyn.pp.recipe_monocle(train_dyn)
    dyn.tl.dynamics(train_dyn, model='stochastic')

    ## compute velocity
    dyn.tl.reduceDimension(train_dyn)
    dyn.tl.cell_velocities(train_dyn, method='pearson', basis='pca', 
                       other_kernels_dict={'transform': 'sqrt'})
    dyn.vf.VectorField(train_dyn, basis='pca', M=1000)

    ## fate bias evaluation
    eval_idx = loadPickle("data/eval_libCB")
    fate_adata = dyn.pd.fate(train_dyn, eval_idx,basis='pca', direction='forward')

    output = dyn.pd.fate_bias(fate_adata, 'Cell type annotation', basis='pca',seed=s)
    output.to_csv(path_or_buf=f'dynamo_fatebias_seed{s}.csv')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed")

    args = parser.parse_args()
    main(args)