import argparse
import configparser
import copy
import json
import os
from datetime import datetime
from json import JSONEncoder
from pathlib import Path
from typing import Dict, List

import anndata
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

import wandb
from marlene.datasets import (
    load_regnetwork,
    load_trrust,
    remove_bad_rows_columns,
    scRNATimeSeriesDataset,
)
from marlene.maml_ct import Meta
from marlene.models.marlene import Marlene
from marlene.utils.evaluation import score


def load_data(config: configparser.ConfigParser):
    # Load data and perform basic filtering
    adata = anndata.read_h5ad(config['data']['path'])
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()

    if config.getboolean('data', 'preprocess'):
        adata = preprocess(adata)

    remove_bad_rows_columns(
        adata,
        celltype=config['data']['celltype'],
        timepoint=config['data']['timepoint'],
        min_cells_per_tp=config.getint('data', 'min_cells_per_tp'),
    )

    trrust_links = load_trrust(
        species=config['data']['species'],
        adata=adata,
    )  # updates adata inplace
    regnetwork_links = load_regnetwork(
        species=config['data']['species'],
        adata=adata,
        index_adata=False,
    )

    n_total_links = adata.shape[1] * adata.var['is_TF'].sum()
    print(f"{n_total_links=}")

    print(f"(cell, genes)={adata.shape}")
    print(f"Total n. TRRUST links={len(trrust_links)}")
    print(f"Total n. RegNetwork links={len(regnetwork_links)}")

    return {
        "adata": adata,
        "n_total_links": n_total_links,
        "TRRUST": trrust_links,
        "RegNetwork": regnetwork_links,
    }


def preprocess(adata):
    """Basic preprocessing following scanpy's tutorial.
    We do this for mouse data only since PBMC data is already preprocessed.

    https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
    """
    print("Preprocessing data using Scanpy")
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_counts=100)
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    return adata


def loss_function(y_pred, y_true):
    loss = F.cross_entropy(y_pred, y_true)
    return loss


def predict_celltype(
    adata: anndata.AnnData,
    celltype: str,
    predictor: nn.Module,
    n_draws: int = 50,
    quantile: float = 0.98,
    tf_mask: np.ndarray | None = None,
) -> List[Dict]:
    """Given a trained model, predict the regulatory links by averaging the
    networks of `n_draws` random batches of cells.

    If tf_mask is given, then will assume that the model output is a square
    matrix and tf_mask is used to select only tfs from that.
    """
    predictor.eval()

    adata_tfs = adata.var_names[adata.var['is_TF']].to_numpy()
    adata_targets = adata.var_names.to_numpy()

    A_seq = np.zeros((dataset.n_timepoints, len(adata_targets), len(adata_tfs)))
    predictor.to(dataset.device)

    with torch.no_grad():
        for _ in range(n_draws):
            x = dataset.sample_celltype_batch(celltype)
            attn = predictor(x, attn_only=True).detach().cpu().numpy()
            if tf_mask is not None:
                attn = attn[..., tf_mask]
            A_seq += attn

    A_seq /= n_draws

    q_seq = [np.quantile(A, quantile) for A in A_seq]

    preds = []  # i-th item is A_i given in edge format

    for A, q in zip(A_seq, q_seq):
        a, b = np.argwhere(A > q).T
        attention_scores = A[a, b]
        tfs, targets = adata_tfs[b], adata_targets[a]
        preds.append({"links": list(zip(tfs, targets)),
                      "attention": attention_scores})

    return preds


class JsonNumpy(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def dump_results(runid, model, best_model, y_pred_seq, config):
    root = Path('Marlene_results') / config['data']['dataset']
    os.makedirs(root / runid, exist_ok=True)
    print(f"Dumping results to '{root / runid}'")

    torch.save(model.state_dict(), root / runid / "model.ckpt")
    torch.save(best_model.state_dict(), root / runid / "best_model.ckpt")

    with open(root / runid / 'y_pred_seq.json', "w") as f:
        json.dump(y_pred_seq, f, indent=4, cls=JsonNumpy)

    with open(root / runid / 'params.ini', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="path to config file", type=str)
    parser.add_argument("--runid", help="ID of this run", type=str,
                        default=f"Marlene-{datetime.now().isoformat()}")
    args = parser.parse_args()

    print(f"Initializing run '{args.runid}'")
    config = configparser.ConfigParser()
    config.read(args.conf)

    data_dict = load_data(config)

    # define dataset
    dataset = scRNATimeSeriesDataset(
        data_dict['adata'],
        scale=config.getboolean('data', 'scale'),
        timepoint_key=config['data']['timepoint'],
        timepoint_order=eval(config['data']['timepoint_order']),
        celltype_key=config['data']['celltype'],
        batch_size=config.getint('data', 'batch_size'),
        device=config['data']['device'],
    )

    # define model
    model = Marlene(
        n_genes=dataset.n_genes,
        n_classes=dataset.n_classes,
        TF_mask=data_dict['adata'].var['is_TF'].to_numpy(),
        n_seeds=config.getint('model', 'n_seeds'),
        n_hidden=config.getint('model', 'n_hidden', fallback=None),
        sparse_q=1 - config.getfloat('model', 'frac_top_edges'),
    ).train()

    if config['data']['device'] == 'cuda':
        model.cuda()

    meta_optimizer = Adam(model.parameters(), lr=config.getfloat('training', 'lr'))

    use_scheduler = config.getboolean('training', 'use_scheduler')
    if use_scheduler:
        scheduler = StepLR(
            meta_optimizer, step_size=config.getint('training', 'scheduler_step_size')
        )

    use_wandb = config.getboolean('training', 'use_wandb')
    if use_wandb:
        run = wandb.init(project="Marlene")
        wandb.watch(model, log='all', log_freq=100)

    meta = Meta(
        model,
        meta_optimizer,
        update_lr=config.getfloat('training', 'inner_lr'),
        update_step=config.getint('training', 'update_step'),
        loss=loss_function,
        gradient_clip=config.getfloat('training', 'gradient_clip'),
    )

    bar = tqdm(range(config.getint('training', 'n_epochs')))

    best_model = None
    best_loss = 1e10
    best_mama = 0

    # Train #################### noqa
    try:
        for epoch in bar:
            batch = dataset.sample_meta_batch()

            x_support_batch = batch['support']
            x_query_batch = batch['query']
            y_support_batch = batch['label']
            y_query_batch = batch['label']

            epoch_loss = 0.0

            epoch_loss, epoch_accuracy = meta(
                x_support_batch, y_support_batch,
                x_query_batch, y_query_batch,
            )

            if epoch_loss < best_loss:  # keep best model
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.cpu())
                if config['data']['device'] == 'cuda':
                    model.cuda()

            if use_scheduler:
                scheduler.step()

            bar.set_description(f"{epoch=}, {epoch_loss=:.4g}")

            if use_wandb:
                wandb.log({
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                    "lr": meta_optimizer.param_groups[0]["lr"],
                })
    except KeyboardInterrupt:
        # save model on keyboard interrupt
        pass

    # Store results ################### noqa
    y_pred_seq = {}
    for celltype in tqdm(dataset.unq_celltype):
        y_pred_seq[celltype] = predict_celltype(
            data_dict['adata'],
            celltype,
            best_model,
            n_draws=config.getint('inference', 'n_draws'),
            quantile=1 - config.getfloat('inference', 'frac_top_edges')
        )

    # Print metrics
    print("RegNetwork quick metrics")
    scores = score(
        y_pred_seq,
        data_dict['RegNetwork'],
        n_total_links=data_dict['n_total_links'],
    )
    print('Mean significant:', (scores['p-val'] < 0.05).mean())
    print('Mean overlap:', (scores['N. overlap']).mean())

    dump_results(args.runid, model, best_model, y_pred_seq, config)
