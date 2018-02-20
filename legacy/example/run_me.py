import sys
from collections import defaultdict
import numpy as np

def default_to_regular(d):
    """Recursively convert nested defaultdicts to nested dicts.
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d


def generate_connected_graph(nodes):
    """Return all possible edges between a set of nodes."""
    edges = set()
    for i1, n1 in enumerate(nodes):
        for i2, n2 in enumerate(nodes):
            if i2 > i1:
                edges.add((n1, n2))
    return edges


def write_pajek(layer_partition):
    """Create pajek file from layer partition."""
    network = "*Vertices "

    nodes = set(
        n
        for l, partition in layer_partition.items()
        for c, nodes in partition.items()
        for n in nodes
    )

    network += str(len(nodes))

    for n in sorted(nodes):
        network += '\n%d "%d" 1.0' % (n, n)

    network += "\n*Intra\n#layer node node [weight]"

    for l, partition in layer_partition.items():
        for c, nodes in partition.items():
            for a, b in generate_connected_graph(nodes):
                network += "\n%d %d %d 1.0" % (l, a, b)

    return network


def two_overlap_benchmark_model(N, m):
    """Create two communities of size `N` that overlap by `m` nodes.

    Two communities each contain `N` nodes and overlap my `m` nodes. All
    nodes in one layer are connected and we stack layers vertically. Examples:

    N=5, m=1:          |  N=5, m=2:        |  N=3, m=2
            o o o o o  |        o o o o o  |    o o o
    o o o o o          |  o o o o o        |  o o o

    Input
    -----
    N : int
        Number of nodes in each community
    m : int
        Number of overlapping nodes between the two communities
    """
    # Create both layers, initially identical
    first_layer = np.arange(0, N)
    second_layer = np.arange(N-m, 2*N - m)

    # Produce multilayer state node label map
    partition = defaultdict(lambda: defaultdict(list))
    for l, nodes in enumerate([first_layer, second_layer]):
        for n in nodes:
            partition[l][l].append(n)

    return default_to_regular(partition)


if __name__ == "__main__":

    # Handle input
    try:
        N, m = int(sys.argv[1]), int(sys.argv[2])
    except IndexError:
        print """Instructions\n------------
run this script with arguments for N (number of nodes in each community),
and m (number of nodes that overlap between the two communities). Try:\n
$ python run_me.py 5 2"""
        sys.exit()

    print "Producing two-layer network with N=%d and m=%d..." % (N, m),

    # Generate two-layer partition
    two_layer_partition = two_overlap_benchmark_model(N, m)

    # Format as pajek
    network = write_pajek(two_layer_partition)

    # Save it
    with open("input/network.net", 'w') as fp:
        fp.write(network)

    print "done!"






















