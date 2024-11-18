from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def score(
    y_pred_seq: Dict[str, List[Dict[str, Any]]],
    true_links: Set[Tuple[str, str]],
    n_total_links: int,
) -> pd.DataFrame:
    """Compare graphs and score overlap with true links.

    Parameters
    ----------
    y_pred_seq: Dict[str, List[Dict[str, Any]]]
        Dictionary mapping a cell type to a list of temporal graphs given
        as lists of links (tuples).
    true_links: Set[Tuple[str, str]]
        Set of links corresponding to known interactions.
    """
    results = []
    true_links = set(true_links)

    for cell_type, preds in y_pred_seq.items():
        graphs = [pred['links'] for pred in preds]

        for t, graph in enumerate(graphs):
            graph: List[Tuple[str, str]]
            overlapping_links = true_links.intersection(graph)
            p = hypergeom.sf(M=n_total_links, n=len(true_links),
                             N=len(graph), k=len(overlapping_links))
            results.append({
                "Cell Type": cell_type,
                "t": t,
                "p-val": p,
                "N. overlap": len(overlapping_links),
                "Overlap": overlapping_links,
                "N. total links": n_total_links,
                "N. true links": len(true_links),
                "N. selected links": len(graph),
            })

    df = pd.DataFrame(results)
    qvals = multipletests(df['p-val'].to_numpy(), method='fdr_bh')[1]
    df['FDR'] = qvals
    df['$-\log_{10}(FDR)$'] = -np.log10(df['FDR'])  # noqa
    return df
