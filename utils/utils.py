from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score

def NMI_score(p1, p2):
    """Takes two partitions in dict format and returns NMI of partition of shared nodes."""
    nodes = sorted(set(p1.keys()) & set(p2.keys()))
    return adjusted_mutual_info_score(
        [p1[n] for n in nodes],
        [p2[n] for n in nodes]
    )

def default_to_regular(d):
    """Recursively convert nested defaultdicts to nested dicts."""
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d


def shuffle_list(l):
    """Non-inline list shuffle.

    Input
    -----
        l : list

    Output
    ------
        out : list
    """
    l_out = list(l)[:]
    np.random.shuffle(l_out)
    return l_out


def shuffle_order(l):
    """Shuffle a list and return the order (for unshuffling) as well."""
    order = shuffle_list(range(len(l)))
    l_shuf = []
    for j in order:
        l_shuf.append(l[j])
    return l_shuf, order


def reorder_shuffled_layer_commu(lc, order):
    """Reorder layer-labels in shuffled multilayer partition.

    Input
    -----
        lc : dict (multilayer partition)
        order : list

    Output
    ------
        out : dict (multilayer partition)
    """
    _layer_commu_ordered = dict()
    for i, j in enumerate(order):
        _layer_commu_ordered[j] = lc[i]
    return _layer_commu_ordered


def unwrap(l, depth=1):
    """Unwrap a list of lists to a single list.

    Input
    -----
        l : list of lists
        depth : number of unwrap operations to perform (int)

    Output
    ------
        out : list
    """
    def _unwrap_one(l):
        return [v for t in l for v in t]
    if depth <= 0:
        return l
    try:
        if depth == 1:
            return _unwrap_one(l)
        return unwrap(_unwrap_one(l), depth=depth-1)
    except TypeError:
        raise TypeError("Max-depth exceeded. Set `depth` to a lower value.")

def invert_partition(partition):
    """Invert a dictionary representation of a graph partition.

    Inverts a dictionary representation of a graph partition from nodes -> communities
    to communities -> lists of nodes, or the other way around.
    """
    if type(partition.items()[0][1]) is list:
        partition_inv = dict()
        for c, nodes in partition.items():
            for n in nodes:
                partition_inv[n] = c
    else:
        partition_inv = defaultdict(list)
        for n, c in partition.items():
            partition_inv[c].append(n)
    return default_to_regular(partition_inv)