import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score

def AMI_score(p1, p2):
    """Takes two partitions in dict format and returns NMI of partition of shared nodes."""
    nodes = sorted(set(p1.keys()) & set(p2.keys()))
    return adjusted_mutual_info_score(
        [p1[n] for n in nodes],
        [p2[n] for n in nodes]
    )

def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A * 1.0 / B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) + _kldiv(Q, M))

def jssim_dist(G1, G2, nodes=None):
    """Get distribution of intra-node Jensen-Shannon similarities.

    Input
    -----
        G1/G2 : nx.Graph
        nodes : list of ints

    Output
    ------
        out : list
    """
    if nodes is None:
        nodes = G1.nodes() | G2.nodes()

    sims = []
    for n in nodes:
        set1, set2 = set(G1.neighbors(n)), set(G2.neighbors(n))
        neighbors = list(set1 | set2)
        p1 = np.array([1./len(set1) if _n in set1 else 0 for _n in neighbors])
        p2 = np.array([1./len(set2) if _n in set2 else 0 for _n in neighbors])
        sims.append(1 - jsdiv(p1, p2))
    
    return sims