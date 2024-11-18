import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class F_SelfAttention(nn.Module):
    """A functional self-attention module.
    """
    def __init__(self, sparse_q: float = 0.98):
        super().__init__()

        self.sparse_q = sparse_q
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: Tensor,
        *,
        query_w: Tensor,
        key_w: Tensor,
        query_b: Tensor | None = None,
        key_b: Tensor | None = None,
        TF_mask: Tensor | None = None,
    ):
        """
        Parameters
        ----------
        x: Tensor of shape (n_timepoints, n_genes, n_seeds)
        query_w, key_w: Tensor of shape (n_seeds, n_seeds)
            Query and Key weights.
        query_q, key_b: Tensor of shape (n_seeds) or None
            Query and Key biases.
        TF_mask: Tensor of shape (n_genes) or None
            True if i-th element is a transcription-factor.
        """
        has_batch = len(x.shape) > 2
        if not has_batch:
            x = x[None]

        input_dim = query_w.shape[0]

        queries = F.linear(x, weight=query_w, bias=query_b)
        if TF_mask is not None:
            keys = F.linear(x[:, TF_mask], weight=key_w, bias=key_b)
        else:
            keys = F.linear(x, weight=key_w, bias=key_b)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (input_dim ** 0.5)

        attention = self.softmax(scores)
        # sparsify
        q = torch.stack([torch.quantile(att, q=self.sparse_q) for att in attention])
        attention = F.relu(attention - q[:, None, None])

        if not has_batch:
            attention = attention[0]

        return attention
