from collections import defaultdict
from typing import Dict, List, Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import normalize as _normalize
from torch.utils.data import Dataset


class scRNATimeSeriesDataset(Dataset):
    def __init__(
        self,
        adata: anndata.AnnData,
        *,
        timepoint_key: str,
        celltype_key: str,
        timepoint_order: List[str] | None = None,
        scale: bool = False,
        normalize: bool = False,
        batch_size: int = 16,
        device: str = 'cuda',
    ):
        """A meta-learning temporal gene dataset.

        Parameters
        ----------
        adata: AnnData
        timepoint_key: str
            Key in adata.obs corresponding to the timepoint for the sample.
        celltype_key: str
            Key in adata.obs corresponding to the cell type for the sample.
        timepoint_order: List[str]
            Order of timepoints in `timepoint_key`. If None, will sort
            unique values in adata.obs[timepoint_key].
        scale: bool
            If True, will apply a standard scaler to X.
        batch_size: int
            Number of samples to draw for each batch.
        device: str
            Device for torch.
        """
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        if scale:
            sc = StandardScaler(with_std=False)
            X = sc.fit_transform(X)
        if normalize:
            X = _normalize(X, norm='l1')

        self.batch_size = batch_size
        self.n_genes = X.shape[1]

        self.timepoint = adata.obs[timepoint_key].to_numpy()
        self.celltype = adata.obs[celltype_key].to_numpy()
        self.timepoint_order = timepoint_order
        if self.timepoint_order is None:
            self.timepoint_order = np.unique(self.timepoint)  # sorts
        assert np.in1d(self.timepoint, self.timepoint_order).all()
        self.n_timepoints = len(self.timepoint_order)

        self.batch_size = batch_size
        self.device = device
        self.unq_celltype = np.unique(self.celltype)
        self.n_classes = self.unq_celltype.size
        self.le = LabelEncoder()
        self.le.fit(self.unq_celltype)

        self.celltype_to_x_seq: Dict[str, List[torch.Tensor]] = {}
        # maps a cell type to [dict['train', 'eval'],  ..., dict] for each timepoint
        self.celltype_to_train_eval_idx: Dict[str, List[Dict[str: np.ndarray]]] = defaultdict(list)

        for CT in self.unq_celltype:
            x_seq: List[torch.Tensor] = []

            for T in self.timepoint_order:
                x = X[(self.timepoint == T) & (self.celltype == CT)]
                x_seq.append(torch.from_numpy(x).float().to(device))
                self.prepare_train_val_split_for_graphs4mer(x, CT)

            self.celltype_to_x_seq[CT] = x_seq

    def __len__(self) -> int:
        """Get number of cell types/tasks"""
        return len(self.unq_celltype)

    def __getitem__(self, index):  # takes all cells, not to be used for training
        if isinstance(index, str):
            celltype = index
        else:
            celltype = self.unq_celltype[index]
        return self.celltype_to_x_seq[celltype], celltype

    def prepare_train_val_split_for_graphs4mer(self, x, CT):
        # for graphs4mer only
        # split data into train a val sets
        train_indices = np.random.choice(x.shape[0], size=x.shape[0] // 2, replace=False)
        train_indices.sort()
        val_indices = np.setdiff1d(np.arange(x.shape[0]), train_indices)

        # if very few cells, mix train or val sets
        # ideally should not need
        if len(train_indices) < self.batch_size:
            train_indices = np.concatenate([
                train_indices,
                np.random.choice(val_indices, self.batch_size - len(train_indices), replace=False)
            ])
        if len(val_indices) < self.batch_size:
            val_indices = np.concatenate([
                val_indices,
                np.random.choice(train_indices, self.batch_size - len(val_indices), replace=False)
            ])

        self.celltype_to_train_eval_idx[CT].append({
            "train": train_indices,
            "val": val_indices
        })

    def subsample(
        self,
        x,
        *,
        nrg: np.random.RandomState | None = None,
        bs: int | None = None,
    ):
        # return a subsample of size batch_size
        batch_size = bs or self.batch_size

        if nrg is not None:
            perm = torch.from_numpy(nrg.permutation(x.size(0)))
        else:
            perm = torch.randperm(x.size(0))
        idx = perm[:batch_size]
        return x[idx]

    def sample_meta_batch(
        self,
        shuffle: bool = True,
        batch_size: int | None = None,
    ):
        x_support_batch, x_query_batch, label_batch = [], [], []

        dist = self.unq_celltype
        if shuffle:
            dist = dist[np.random.permutation(len(dist))]

        for label in dist:
            x_seq = self.celltype_to_x_seq[label]  # List[Tensor]
            x_support_batch.append(torch.stack(
                [self.subsample(x, bs=batch_size) for x in x_seq]
            ))
            x_query_batch.append(torch.stack(
                [self.subsample(x, bs=batch_size) for x in x_seq]
            ))
            label_batch.append(label)

        return {
            "support": torch.stack(x_support_batch),  # List[Tensor()]
            "query": torch.stack(x_query_batch),
            "label": torch.from_numpy(self.le.transform(label_batch)).long().to(self.device),
        }

    def sample_celltype_batch(
        self,
        label: str,
        nrg: np.random.RandomState | None = None,
        stage: Literal['train', 'val'] | None = None,
        batch_size: int | None = None,
    ):
        x_seq = self.celltype_to_x_seq[label]
        if stage is not None:
            # take train or eval samples only. only for graphs4mer
            idx_seq = self.celltype_to_train_eval_idx[label]
            x_seq = [x[idx[stage]] for x, idx in zip(x_seq, idx_seq)]
        return torch.stack([self.subsample(x, nrg=nrg, bs=batch_size) for x in x_seq])


def remove_bad_rows_columns(
    adata: anndata.AnnData,
    celltype: str,
    timepoint: str,
    min_cells_per_tp: int = 500,
    res_genes: np.ndarray | None = None,
) -> None:
    # Remove cell types with few cells in any timepoint
    _cts = adata.obs[celltype].to_numpy()
    _tps = adata.obs[timepoint].to_numpy()

    to_remove = []
    for tp in np.unique(_tps):
        for ct in np.unique(_cts):
            if ((_tps == tp) & (_cts == ct)).sum() < min_cells_per_tp:
                to_remove.append(ct)
    to_remove = np.unique(to_remove)

    n_celltypes = adata.obs[celltype].unique().size
    print(f"Removing {len(to_remove)} low count cell types")
    print(f"Using {n_celltypes - len(to_remove)} cell types")
    cell_mask = ~np.in1d(adata.obs[celltype], to_remove)
    cell_mask[pd.isna(adata.obs[celltype])] = False

    adata._inplace_subset_obs(cell_mask)

    to_keep = np.ones(adata.shape[1], dtype=bool)
    if res_genes is not None:
        to_keep[~np.in1d(adata.var_names, res_genes)] = False

    # Remove genes which are zero everywhere for any timepoint
    for tp in adata.obs[timepoint].unique():
        is_nonzero = np.asarray(
            (adata[adata.obs[timepoint] == tp].X != 0).sum(0)
        ).ravel() != 0
        to_keep &= is_nonzero
    print(f"Removing {to_keep.size - to_keep.sum()} zero genes")
    adata._inplace_subset_var(to_keep)


def load_trrust(
    species: Literal['human', 'mouse'] = 'human',
    adata: anndata.AnnData | None = None,
    index_adata: bool = True,
):
    """Loads the TRRUST database. If adata is provided and index_adata
    is True, will also remove non-overlapping genes from adata."""
    path = f'data/trrust_rawdata.{species}.tsv'
    trrust = pd.read_csv(path, sep='\t', header=None)
    if adata is None:
        return trrust

    trrust[0] = trrust[0].str.upper()
    trrust[1] = trrust[1].str.upper()

    adata.var_names = np.char.upper(adata.var_names.to_numpy().astype(str))

    # Keep only links where both the TF and target can be found in adata
    trrust = trrust[(np.in1d(trrust[0], adata.var_names))
                    & (np.in1d(trrust[1], adata.var_names))]
    trrust_links = list(set(zip(trrust[0], trrust[1])))

    all_unq_trrust_genes = pd.unique(trrust[[0, 1]].values.ravel('K'))
    common_genes = np.intersect1d(all_unq_trrust_genes, adata.var_names)
    print(f"Found {len(common_genes)} genes in common.")
    if index_adata:
        adata._inplace_subset_var(common_genes)
        adata.var['is_TF'] = np.in1d(adata.var_names, trrust[0].unique())
        print(f"Using {adata.var['is_TF'].sum()} TFs")
    return trrust_links


def load_regnetwork(
    species: Literal['human', 'mouse'] = 'human',
    adata: anndata.AnnData | None = None,
    index_adata: bool = True,
):
    """Loads the RegNetwork database. If adata is provided and index_adata
    is True, will also remove non-overlapping genes from adata."""
    path = f'data/RegNetwork-{species}.source'
    regnetwork = pd.read_csv(path, sep='\t', header=None)
    regnetwork = regnetwork[~(
        (regnetwork[0].str.startswith('hsa'))
        | (regnetwork[2].str.startswith('hsa'))
        | (regnetwork[0].str.startswith('mnu'))
        | (regnetwork[2].str.startswith('mnu'))
    )]
    if adata is None:
        return regnetwork

    regnetwork[0] = regnetwork[0].str.upper()
    regnetwork[2] = regnetwork[2].str.upper()

    adata.var_names = np.char.upper(adata.var_names.to_numpy().astype(str))

    tf_names = (
        adata.var_names if 'is_TF' not in adata.var
        else adata.var_names[adata.var['is_TF']]
    )
    regnetwork = regnetwork[
        (np.in1d(regnetwork[0], tf_names))
        & (np.in1d(regnetwork[2], adata.var_names))
    ]

    regnetwork_links = list(set(zip(regnetwork[0], regnetwork[2])))

    all_unq_regnetwork_genes = pd.unique(regnetwork[[0, 2]].values.ravel('K'))
    common_genes = np.intersect1d(all_unq_regnetwork_genes, adata.var_names)
    print(f"Found {len(common_genes)} genes in common.")
    if index_adata:
        adata._inplace_subset_var(common_genes)
        adata.var['is_TF'] = np.in1d(adata.var_names, regnetwork[0].unique())
        print(f"Using {adata.var['is_TF'].sum()} TFs")
    return regnetwork_links
