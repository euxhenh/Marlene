from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .attention import F_SelfAttention
from .set_transformer import PMA


class Marlene(nn.Module):
    """Meta learning for temporal gene regulatory networks.

    Parameters
    ----------
    n_genes: int
        Number of genes.
    n_classes: int
        Number of classes, i.e., cell types in the data.
    n_hidden: int, None
        If not None, will add a relu and another output linear layer at the
        end of the network that takes n_hidden dimensions.
    TF_mask: np.ndarray, None
        A mask of transcription-factors. If None, will learn (n_genes,
        n_genes) adjacency matrices. If not None, will learn (n_genes,
        TF_mask.sum()).
    n_seeds: int
        Number of gene features to obtain using PMA.
    sparse_q: int
        Quantile to use for selecting edges. Will only keep the top
        (1-sparse_q) edges.
    """
    if TYPE_CHECKING:
        TF_mask: torch.BoolTensor

    def __init__(
        self,
        *,
        n_genes: int,
        n_classes: int,
        n_hidden: int | None = None,
        TF_mask: np.ndarray | None,
        n_seeds: int = 16,
        sparse_q: float = 0.98,
    ):
        super().__init__()
        if TF_mask is not None:
            TF_mask = torch.Tensor(TF_mask).bool()

        self.register_buffer('TF_mask', TF_mask)

        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_seeds = n_seeds

        self.pma = PMA(dim=n_genes, num_heads=1, num_seeds=n_seeds)

        self.attention = F_SelfAttention(sparse_q=sparse_q)
        self.query_w = nn.Parameter(torch.Tensor(n_seeds, n_seeds))
        self.key_w = nn.Parameter(torch.Tensor(n_seeds, n_seeds))

        self.gru_query = nn.GRU(n_seeds, n_seeds)
        self.gru_key = nn.GRU(n_seeds, n_seeds)

        self.topk = TopK(n_seeds, n_seeds)

        if n_hidden is None:
            self.lin1 = nn.Linear(n_genes, n_classes)
        else:
            self.lin1 = nn.Linear(n_genes, n_hidden)
            self.relu = nn.ReLU()
            self.lin2 = nn.Linear(n_hidden, n_classes)

        self.reset_params()

    def reset_params(self) -> None:
        xavier_uniform_(self.query_w)
        xavier_uniform_(self.key_w)

    def forward(self, x_seq: List[torch.Tensor], attn_only: bool = False):
        """
        Parameters
        ----------
        x_seq: Tensor of shape (n_timepoints, batch_size, n_genes)
            batch_size is the number of cells in a batch.

        attn_only: bool
            If True, will return attention only

        Returns
        -------
        out_seq: Tensor of shape (n_classes,)
            The cell type prediction for the batch.
        attn_seq: Tensor of shape (n_timepoints, n_genes, n_tfs)
            These adj matrices are in target-to-source mode.
        """
        h_seq = self.pma(x_seq)  # (n_timepoints, n_seeds, n_genes)
        h_seq = torch.moveaxis(h_seq, 1, 2)  # (n_timepoints, n_genes, n_seeds)

        query_w = self.query_w
        key_w = self.key_w

        attn_seq = []

        for h in h_seq:
            h_pool = self.topk(h)

            # evolve weights
            _, query_w = self.gru_query(h_pool[None], query_w[None])
            _, key_w = self.gru_key(h_pool[None], key_w[None])
            query_w, key_w = query_w[-1], key_w[-1]
            assert query_w.shape == key_w.shape == (self.n_seeds, self.n_seeds)

            A = self.attention(x=h, query_w=query_w, key_w=key_w, TF_mask=self.TF_mask)
            attn_seq.append(A)

        attn_seq = torch.stack(attn_seq)  # (n_timepoints, n_genes, n_tfs)

        if attn_only:
            return attn_seq

        if self.TF_mask is not None:
            x_seq = x_seq[..., self.TF_mask]  # (n_timepoints, n_cells, n_tfs)

        outs = []
        for x, A in zip(x_seq, attn_seq):
            out = self.lin1(x @ A.T)  # (n_cells, n_classes)
            if self.n_hidden is not None:
                out = self.relu(out)
                out = self.lin2(out)
            out = out.sum(0)  # sum along cells, (n_classes,)
            outs.append(out)

        # sum along timepoints
        out = torch.sum(torch.stack(outs), dim=0)  # (n_classes,)
        assert out.shape == (self.n_classes,)

        return out, attn_seq


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


class TopK(torch.nn.Module):
    """
    Code taken from https://github.com/IBM/EvolveGCN.
    """
    def __init__(self, n_features, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(n_features, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / np.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        return out.t()
