from collections import Counter, defaultdict
import numpy as np

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

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )