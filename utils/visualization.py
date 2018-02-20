import utils

import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import networkx as nx
import community

def standarize_plot_parameters():
    # http://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/LaTeX_Examples
    # thesis has 417.47 points in column size, with 0.6\columnwidth
    fig_width_pt = 417.47*0.6
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean       # height in inches
    
    params = {
        'axes.labelsize': 10,
        'legend.fontsize': 7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'STIXGeneral',  # close enough to LaTeX font
        'font.size': 8,
        'figure.frameon': False
    }
    plt.rcParams.update(params)


def confidence_intervals(X, Y, c, label="", mid_50_percentile=False, lw=2, ls="-"):
    X, Y = np.array(X), np.array(Y)
    low, upp = st.t.interval(
        0.99,
        Y.shape[0]-1,
        loc=np.mean(Y, axis=0),
        scale=st.sem(Y)
    )
    plt.fill_between(
        np.mean(X, axis=0),
        low,
        upp,
        alpha=0.5,
        color=c,
        lw=0
    )
    plt.plot(
        np.mean(X, axis=0),
        np.mean(Y, axis=0),
        lw=lw,
        ls=ls,
        c=c,
        label=label
    )
    if mid_50_percentile:
        plt.fill_between(
            np.mean(X, axis=0),
            np.percentile(Y, 25, axis=0),
            np.percentile(Y, 75, axis=0),
            alpha=.25,
            color=c
        )


def draw(G, partition=False, colormap='rainbow', labels=None):
    """Draw graph G in my standard style.

    Input
    -----
    G : networkx graph
    partition : bool
    colormap : matplotlib colormap
    labels : dict (Node labels in a dictionary keyed by node of text labels)
    """
    
    def _get_cols(partition):
        return dict(
            zip(
                utils.shuffle_list(set(partition.values())),
                np.linspace(0, 256, len(set(partition.values()))).astype(int)
            )
        )

    cmap = plt.get_cmap(colormap)
    if partition == True:
        partition = community.best_partition(G)
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) is dict and len(partition) >= len(G.nodes()):
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) in [list, tuple] and len(partition) == len(G.nodes()):
        colors = list(partition)
    else:
        try:
            colors = [n[1]['node_color'] for n in G.nodes(data=True)]
        except KeyError:
            # nodes do not have node_color attribute
            colors = "grey"
    
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    nx.draw_networkx_edges(G, pos=pos, width=2, alpha=.3, zorder=-10)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, alpha=1, linewidths=0, node_color=colors)
    
    if labels is not None:
        nx.draw_networkx_labels(G, pos=dict((k, (v[0]+15, v[1])) for k, v in pos.items()), labels=labels, font_size=16)

    plt.axis("off")