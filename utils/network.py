import utils

import subprocess
import re, os
from collections import defaultdict, Counter
import networkx as nx

def Infomap(pajek_string, *args, **kwargs):
    """Function that pipes commands to subprocess and runs native Infomap implementation.
    
    Requires a root /tmp directory which can be written to, as well as an Infomap executable
    in the /usr/local/bin. Works for version 0.18.25 and higher.
    
    Parameters
    ----------
    pajek_string : str
        Pajek representation of the network (str)
    *args : dict
        Infomap execution options. (http://www.mapequation.org/code.html#Options)
        
    Returns
    -------
    communities : dict of Counters
    layer_communities : dict of dicts (layer, layer communities, community members)
    node_flow : dict
    community_flow : dict

    Example
    -------
    >>> network_pajek = write_pajek(multilayer_edge_list)
    >>>
    >>> communities, layer_communities, node_flow, community_flow = Infomap(
    >>>     network_pajek,
    >>>     '-i', 'multiplex',
    >>>     '--multiplex-js-relax-rate', '0.25',
    >>>     '--overlapping',
    >>>     '--expanded',  # required
    >>>     '--clu',       # required
    >>>     '-z',          # required if multiplex
    >>>     '--two-level'
    >>> )

    """
    
    def _get_id_to_label(filename):
        def __int_if_int(val):
            try: return int(val)
            except ValueError: return val
        with open(home + '/tmp/input_infomap/' + filename + ".net", 'r') as fp:
            parsed_network = fp.read()
        return dict(
            (int(n.split()[0]), __int_if_int(n.split('"')[1]))
            for n in re.split(r"\*.+", parsed_network)[1].split("\n")[1:-1]
        )
    
    def _parse_communities_multiplex(id_to_label, filename):
        with open(home + '/tmp/output_infomap/'+filename+"_expanded.clu", 'r') as infile:
            clusters = infile.read()

        # Get layers, nodes and clusters from _extended.clu file
        la_no_clu_flow = re.findall(r'\d+ \d+ \d+ \d.*\d*', clusters) # ["30 1 2 0.00800543",...]
        la_no_clu_flow = [tuple(i.split()) for i in la_no_clu_flow]

        node_flow_json = defaultdict(float)      # {layer_node: flow, ...}
        community_flow_json = defaultdict(float) # {community: flow, ...}
        communities_json = defaultdict(set)      # {layer: {(node, cluster), ...}, ...}
        for layer, node, cluster, flow in la_no_clu_flow:
            node_flow_json["%s_%s" % (layer, id_to_label[int(node)])] += float(flow)
            community_flow_json[cluster] += float(flow)
            communities_json[int(layer)].add((id_to_label[int(node)], int(cluster)))

        return communities_json, node_flow_json, community_flow_json
    
    def _parse_communities_planar(id_to_label, filename):
        with open(home + '/tmp/output_infomap/'+filename+".clu", 'r') as infile:
            clusters = infile.read()
        
        # Get nodes and clusters from .clu file
        no_clu = [tuple(i.split()[:-1]) for i in re.findall(r"\d+ \d+ \d.*\d*", clusters)]  # [(node, cluster), ...]
        return {0: set([(id_to_label[int(no)], int(clu)) for no, clu in no_clu])}
    
    def _clean_up(filename):
        subprocess.call(['rm', home + '/tmp/input_infomap/' + filename + '.net'])
        subprocess.call(['rm', home + '/tmp/output_infomap/' + filename + '_expanded.clu'])
        subprocess.call(['rm', home + '/tmp/output_infomap/' + filename + '.clu'])
    
    # Check for process id in args (for multiprocessing)
    if args[-1][:3] == "pid":
        pid = args[-1][3:]
        args = args[:-1]
    else:
        pid = ""
        
    # Get user home directory
    home = os.path.expanduser("~")

    # Try to make input_infomap and output_infomap folders inhome +  /tmp
    subprocess.call(['mkdir', home + '/tmp/input_infomap', home + '/tmp/output_infomap'])
    
    # Get network in multiplex string format and define filename
    filename = 'tmpnet' + pid

    # Store locally
    with open(home + "/tmp/input_infomap/"+filename+".net", 'w') as outfile:
        outfile.write(pajek_string)
    
    # Run Infomap for multiplex network
    subprocess.call(
        ['Infomap', home + '/tmp/input_infomap/'+filename+".net", home + '/tmp/output_infomap'] + \
        list(args)
    )
    
    # Parse communities from Infomap output_infomap
    id_to_label = _get_id_to_label(filename)
    
    if 'multiplex' in list(args):
        parsed_communities, node_flow, community_flow = _parse_communities_multiplex(id_to_label, filename)
    if 'pajek' in list(args):
        parsed_communities = _parse_communities_planar(id_to_label, filename)
        
    _clean_up(filename)

    # Produce layer communities
    layer_communities = {}
    for layer, group in parsed_communities.items():
        communities = {}
        for no, clu in group: 
            try:
                communities[clu-1].append(no)
            except KeyError:
                communities[clu-1] = [no]
        layer_communities[layer] = communities
        
    # Produce community_members
    community_members = defaultdict(Counter)
    for _, communities in layer_communities.items():
        for c, members in communities.items():
            community_members[c].update(members)

    return [
        utils.default_to_regular(community_members),
        layer_communities,
        utils.default_to_regular(node_flow),
        utils.default_to_regular(community_flow)
    ]

def write_pajek(ml_edgelist, index_from=0):
    """Return multiplex representation of multiplex network adjacency matrix A.

    Providing an adjacency tensor where A[:, :, k] is adjacency matrix of layer k, 
    return a pajek format representation of the multilayer network which weights interlayer
    edges by state node neighborhood similarity. 

    Parameters
    ----------
    ml_edgelist : pd.DataFrame
        Must have the three columns `node1`, `node2` and `layer`
    index_from : int
        From which number to index nodes and layers in pajek format from (default=0)

    Returns
    -------
    out : string
        A network string in pajek format
    """
    def _build_adjacency_tensor(ml_edgelist, index="zero"):
        """Return adjacency tensor representation of multilayer edgelist."""
        layers = sorted(set(ml_edgelist['layer']))
        nodes = set(list(ml_edgelist['node1']) + list(ml_edgelist['node2']))
        ind = dict((n, i) for i, n in enumerate(nodes))

        A = defaultdict(int)
        for l in layers:
            for _, row in ml_edgelist.loc[ml_edgelist['layer'] == l].iterrows():
                # Must add both ways if undirected so A becomes symmetrical. If only added one-way
                # triu will only be connections from 'node1' and and tril from 'node2' or vice versa.
                if index == "zero":
                    A[(ind[row['node1']], ind[row['node2']], l)] += 1
                    A[(ind[row['node2']], ind[row['node1']], l)] += 1
                else:
                    A[(row['node1'], row['node2'], l)] += 1
                    A[(row['node2'], row['node1'], l)] += 1
        return A, dict((v, k) for k, v in ind.items())

    def _write_outfile(A):
        """Write nodes and intra/inter-edges from A and J to string."""
        def __remove_symmetry_A(A):
            A_triu = defaultdict(int)
            for (i, j, k), w in A.items():
                if j > i:
                    A_triu[(i, j, k)] = w
            return A_triu
        def __write_nodes(outfile):
            outfile += "*Vertices %d" % Nn
            for nid, label in enumerate(nodes):
                outfile += '\n%d "%s" 1.0' % (nid + index_from, labelmap[label])
            return outfile
        def __write_edges(outfile):
            outfile += "\n*Intra\n# layer node node [weight]"
            sorted_A_sparse = sorted(__remove_symmetry_A(A).items(), key=lambda (ind, _): ind[2])
            for (i, j, k), w in sorted_A_sparse:
                outfile += '\n%d %d %d %f' % (
                    k + index_from,  # layer
                    nodemap[i] + index_from,  # node
                    nodemap[j] + index_from,  # node
                    w                         # weight
                )
            return outfile
        
        outfile = ""
        outfile = __write_nodes(outfile)
        outfile = __write_edges(outfile)
        
        return outfile

    A, labelmap = _build_adjacency_tensor(ml_edgelist)

    nodes = sorted(set([n for i, j, _ in A.keys() for n in [i, j]]))
    Nn = len(nodes)
    Nl = len(set([k for i, j, k in A.keys()]))

    nodemap = dict(zip(nodes, range(Nn)))

    return _write_outfile(A)


def convert_to_pajek(layer_partition, sparse_frac=1):
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
            links = generate_connected_graph(nodes)
            for a, b in utils.shuffle_list(links)[:int(len(links)*sparse_frac)]:
                network += "\n%d %d %d 1.0" % (l, a, b)

    return network


def generate_connected_graph(nodes):
    """Return all possible edges between a set of nodes."""
    edges = set()
    for i1, n1 in enumerate(nodes):
        for i2, n2 in enumerate(nodes):
            if i2 > i1:
                edges.add((n1, n2))
    return edges


def graph_list_to_pajek(G_list):
    """Convert list of graphs to multilayer pajek string
    
    Input
    -----
    G_list : list
        Networkx graphs
    
    Output
    ------
    out : str
        Pajek filestring in *Intra format
    """
    def _write_pajek(A, node_labels=None, index_from=0):
        """Return multiplex representation of multiplex network adjacency matrix A

        Providing an adjacency tensor where A[:, :, k] is adjacency matrix of temporal
        layer k, return a pajek format representation of the temporal network which weights interlayer
        edges by state node neighborhood similarity. 

        Parameters
        ----------
        A : numpy.3darray
            3d tensor where each A[:, :, k] is a layer adjacency matrix
        max_trans_prob : float/str
            Cap on interlayer edge weights. 'square' for square penalty.
        power_penalty : int/float
            Power to jaccard similarity betw. state nodes to penalize low similarity
        index_from : int
            From which number to index nodes and layers in pajek format from
        style : bool
            Either 'zigzag', 'vertical', or 'simple'. 'vertical' will give working results but is
            essentially wrong use of Infomap, 'simple' should be possible to use in Infomap but is not
            at this point, so 'zigzag' is preferred because it is an explicit representation of the way
            the network should be represented internally in Infomap.

        Returns
        -------
        out_file : string
            A network string in multiplex format
        intid_to_origid : dict
            Key-value pairs of node integer id and original id
        origid_to_intid : dict
            Reverse of intid_to_origid
        """

        def _write_outfile(A):
            """Write nodes and intra/inter-edges from A and J to string."""
            def __remove_symmetry_A(A):
                A_triu = defaultdict(int)
                for (i, j, k), w in A.items():
                    if j > i:
                        A_triu[(i, j, k)] = w
                return A_triu
            def __write_nodes(outfile):
                outfile += "*Vertices %d" % Nn
                for nid, label in enumerate(nodes):
                    outfile += '\n%d "%s" 1.0' % (nid + index_from, str(label))
                return outfile
            def __write_intra_edges(outfile):
                outfile += "\n*Intra\n# layer node node [weight]"
                for (i, j, k), w in __remove_symmetry_A(A).items():
                    outfile += '\n%d %d %d %f' % (
                        k + index_from,  # layer
                        nodemap[i] + index_from,  # node
                        nodemap[j] + index_from,  # node
                        w                # weight
                    )
                return outfile

            outfile = ""
            outfile = __write_nodes(outfile)
            outfile = __write_intra_edges(outfile)

            return outfile

        nodes = sorted(set([n for i, j, _ in A.keys() for n in [i, j]]))
        Nn = len(nodes)
        Nl = len(set([k for i, j, k in A.keys()]))

        nodemap = dict(zip(nodes, range(Nn)))

        return _write_outfile(A)

    def _create_adjacency_matrix(layer_edges):
        """Return 3d adjacency matrix of the temporal network.
        
        Input
        -----
        layer_edges : dict
        
        Output
        ------
        A : dict
        """
        A = defaultdict(int)
        for l, edges in layer_edges.items():
            for edge in edges:
                    A[(edge[0], edge[1], l)] += 1
                    A[(edge[1], edge[0], l)] += 1    
        return A
    
    return _write_pajek(
        _create_adjacency_matrix(
            dict(zip(range(len(G_list)), [G.edges() for G in G_list]))
        )
    )


def LFR_benchmark_graph(N, k, maxk, mu,  t1=None, t2=None, minc=None, maxc=None, on=None, om=None, C=None):
    """Produce LFR benchmark network from original C++ code.
    
    Source: https://sites.google.com/site/andrealancichinetti/files.
    Requires that `binary_networks` is installed (e.g. in usr/local/bin) and
    globally referencable as `benchmark`.
    
    Input
    -----
        N : number of nodes
        k : average degree
        maxk : maximum degree
        mu : mixing parameter
        t1 : minus exponent for the degree sequence
        t2 : minus exponent for the community size distribution
        minc : minimum for the community sizes
        maxc : maximum for the community sizes
        on : number of overlapping nodes
        om : number of memberships of the overlapping nodes
        C : average clustering coefficient
        
    Output
    ------
        G : nx.Graph
        community : dict (partition)
    """
    # Run program
    arguments = ["benchmark", "-N", str(N), "-k", str(k), "-maxk", str(maxk), "-mu", str(mu)]
    extra_input_args = [("t1", t1), ("t2", t2), ("minc", minc), ("maxc", maxc), ("on", on), ("om", om), ("C", C)]
    extra_arguments = utils.unwrap([("-%s" % k, str(v)) for k, v in extra_input_args if v is not None])
    subprocess.call(arguments + extra_arguments)

    # Collect results
    network = set()
    with open("network.dat") as fp:
        for line in fp:
            a, b = sorted(map(int, re.findall(r"\d+", line)))
            network.add((a, b))

    community = {}
    with open("community.dat") as fp:
        for line in fp:
            a, b = map(int, re.findall(r"\d+", line))
            community[a] = b
            
    G = nx.Graph()
    G.add_edges_from(network)
    
    return G, community
    plt.hist([d for n, d in G.degree()], bins=100)