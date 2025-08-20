import torch
import numpy as np

from beyondicnn.utils_edge_subdiv import get_unit_hypercube
from beyondicnn.utils_edge_subdiv import get_labels

# from https://github.com/arturs-berzins/relu_edge_subdivision

# NOTE: important for consistency between cpu and cuda https://github.com/pytorch/pytorch/issues/77397
torch.backends.cuda.matmul.allow_tf32 = False


def get_intersecting_via_perturb(em_sv, d):
    '''
    Find intersecting edges (face, hyperplane intersections).
    Take all splitting edges. For each splitting edge find all incident faces by perturbing the sign-vector.
    This means taking one zero at a time and setting it to + or -. Since an edge has (d-1) zeros,
    this gives 2*(d-1) faces incident to each edge. Now we find the unique faces and the two edges
    that point to it. This is the only expensive step.
    Memory is O(e*2*(d-1)) where e is the nof split edges. Much better than O(n^2) when doing pairwise adjacency check.
    Output:
    novel_idxs_of_adj_edges is shape [#nof splitting edges, 2].
    novel_idxs_of_adj_edges[e] gives two indices i,j.
    This are the indices of edges in the _splitting_ edge array i.e. _masked_ edge array.
    E.g. e_tb[:,split_edge_mask][i] is the tb of the first edge.
    Convert to index in the full edge array via split_edge_mask.nonzero()[:,0][i]
    '''
    device = em_sv.device
    # Range to adress (2)*(d-1) faces incident to edges
    rnge_dim = torch.arange(d-1, dtype=torch.int64, device=device)

    # No splitting edges, return an empty pair
    if len(em_sv) == 0:
        return torch.empty(0, 2, dtype=int, device=device)

    ### Perturb edge sign-vectors to find faces ###

    # Repeat each edge d-1 times for each zero and 2 times for +- which it can become.
    incident_faces_sv = em_sv[None, ...].repeat(2, d-1, 1, 1)

    # Find the locations of the zeros, which we will change to +-
    zero_idxs = (em_sv == 0).nonzero()
    # Create a tuple of indices for advanced indexing (analogous to what nonzero would return):
    # 1st adresses all d-1 zeros w/ range
    # 2nd adresses row locations of zeros (as in over perturbed edges)
    # 3rd adresses col locations of zeros (as in over constraints)
    # print(zero_idxs.shape, len(em_sv))
    if not len(zero_idxs) == len(em_sv)*(d-1):
        print("not generic  !")
        # TODO: handle non-generic
        return torch.empty(0, 2, dtype=int, device=device)

    # "The new hyperplane intersects an old vertex, the arrangement is not generic"
    idxs = (rnge_dim.repeat_interleave(len(em_sv)),) \
        + tuple(zero_idxs.reshape(len(em_sv), d-1,
                2).permute(1, 0, 2).flatten(end_dim=1).T)

    # Perturb to +-. TODO: perturb boundary cells only toward + to make b a bit smaller. Not sure if modulo will work though.
    incident_faces_sv[0][idxs] = 1
    incident_faces_sv[1][idxs] = -1

    # Make a long list (a 2D tensor) of all face sign-vectors. Permute so we can use modulo in the end.
    b = incident_faces_sv.permute(1, 0, 2, 3).flatten(end_dim=2)

    ### Find unique faces and their edges ###
    if device.type == 'cuda':
        # On cuda unique is faster: find unique faces and the indices telling where each face (perturbed edge) from b is in unique faces.
        _, inv = b.unique(dim=0, return_inverse=True)
        # If we sort the inv, the perturbed edge pairs from b will be together in the sorted list.
        inv_sorted, inv_inds = inv.sort()
        # Two same subsequent entries indicate a splitting face and a face-novel edge
        idxs_of_edge_pairs_in_inv = (
            ~inv_sorted.diff().bool()).nonzero()  # Maybe bool mask?
        # Now, indices of both perturbed edges sharing the face are offset by one. Convert these indices into b array indexing via inv_inds.
        # Lastly, find indices of the edge pairs in the previous edge list by reversing the repeat with modulo.
        novel_idxs_of_adj_edges = inv_inds[torch.hstack(
            [idxs_of_edge_pairs_in_inv, idxs_of_edge_pairs_in_inv+1])] % len(em_sv)
    else:
        # On cpu lexsort is faster. Part of the reason is that pytorch cuda just does not have a native lexsort.
        # The best available port is decent, but still worse than the native unique. This might be different with JAX.
        # https://dagshub.com/safraeli/attention-learn-to-route/src/674e5760ce82183a56c94f453aaaf37fdf8e1953/utils/lexsort.py
        b_lex_idxs = torch.from_numpy(np.lexsort(b.numpy().T))
        b_lex = b[b_lex_idxs]
        idxs_of_edge_pairs_in_lex = torch.all(
            ~b_lex.diff(dim=0).bool(), dim=1).nonzero()[:, 0]
        novel_idxs_of_adj_edges = b_lex_idxs[torch.vstack(
            [idxs_of_edge_pairs_in_lex, idxs_of_edge_pairs_in_lex+1]).T] % len(em_sv)

    # em_sv[novel_idxs_of_adj_edges] are the sign-vectors of pairs of adjacent edges.
    # They will have a single common 0 entry and D different entries where one is 0 at he other is +-.
    # All other entries are identical +-.
    # ##print(em_sv[novel_idxs_of_adj_edges])
    # These indices are for the masked edge array of splitting edges
    return novel_idxs_of_adj_edges


def get_e_sv(v_sv, edges):
    """Build edge sign-vectors from vertex sign-vectors."""
    # ##print("vs_edges", v_sv[edges])
    # sum over activations' pairs
    return v_sv[edges].sum(1, dtype=torch.int8).sign()


def lin_interp(x1, x2, y1, y2):
    """In linear inteprolation
    - xs are the neuron values
    - ys are the vertex coordinates"""
    return y1 - x1*(y2-y1)/(x2-x1)


def skeletal_subdivision(f, bbox=(-1, 1), device=None, verbose=True, plot=False, return_intermediate=False, return_memory=False, allow_non_generic=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and return_memory:
        torch.cuda.reset_peak_memory_stats()

    # Move the NN to the device
    f.to(device)

    # Dimension of input
    d = f.ks[0]

    # Init hypercube
    # VS_BITS = COORDINATES, EDGES = [[IDX_VERTEX_1, IDX_VERTEX_2]]
    vs_bits, edges = get_unit_hypercube(d)
    vs_bits = vs_bits.to(device)
    edges = edges.to(device)

    # sign vertices
    v_sv = torch.hstack([vs_bits, ~vs_bits]).to(dtype=torch.int8)
    # vertices
    vs = (vs_bits*(bbox[1]-bbox[0]) + bbox[0]).float()
    # print("v_sv", v_sv)
    del vs_bits

    if return_intermediate:
        intermediates = {}

    # N=0
    v_mask = None
    lv = 0
    vertices_vals_for_all_neurons = torch.empty(
        0, sum(f.ks[1:]), device=device)

    # Go through layers
    total_depth = len(f.ks)
    for l in range(1, total_depth-1):
        K = sum(f.ks[1:l])  # hidden neurons already considered
        # Go through neurons
        for k in range(f.ks[l]):  # f.ks[l] = number of neurons for layer l
            # print("LAYER", l, "NEURON", k)

            # evaluate vertices
            # has dim = n_vertices x n_neurons (without input neurons), computes pre-activation of each neuron on each vertex
            # lv keeps track of already computed vertices, new vertices are added at the end of lv
            vertices_vals_for_all_neurons = torch.vstack(
                [vertices_vals_for_all_neurons, f.eval_block(vs[lv:], device=device)])
            # Get values w.r.t. this neuron
            vals = vertices_vals_for_all_neurons[:, K+k]
            # print("vals", vals)

            if torch.any(vals == 0):  # torch.abs(vals)<1e-8
                # #print(
                # "Warning: a hyperplane (corresponding to neuron {} in layer {} intersects a vertex (up to numerical precision)".format(k, l))
                # #print(
                # "The arrangement is not generic. 0 will be treated as +. This may cause unexpected behaviour.")
                assert allow_non_generic

            # Get the sign of old vertices
            old_vert_new_sv = torch.sign(vals).to(dtype=torch.int8)[:, None]
            old_vert_new_sv[old_vert_new_sv == 0] = 1  # to handle non-generic

            # (2,3) Compare vertex pair signs
            vals_pairs = vals[edges]
            # print("vals_pairs", vals_pairs)
            del vals

            # Find splitting edges by looking at signs of their vertices
            splitting_edge_mask = vals_pairs.sign().prod(1) == -1

            # I.E. if a pair of connected vertices has different sign in vals, there is an hyperplane crossing the edge

            # (4) Interpolate to find new vertices on splitting edges
            # f(x_1)= a>0, f(x_2)= b< 0
            # if x s.t. f(x)=0, then a / (x_2 -x) = b / (x - x_1) in 1D.
            #  Same in dD. Enables to compute x.
            new_vertices = lin_interp(
                vals_pairs[splitting_edge_mask, None, 0],
                vals_pairs[splitting_edge_mask, None, 1],
                vs[edges][splitting_edge_mask, 0],
                vs[edges][splitting_edge_mask, 1])
            # print("new_vertices", new_vertices)
            del vals_pairs
            ### Intersecting edges ###
            # #print(v_sv, edges[splitting_edge_mask])
            em_sv = get_e_sv(v_sv, edges[splitting_edge_mask])
            # #print("em_sv", em_sv)
            # print("em_sv", em_sv)

            novel_idxs_of_adj_edges = get_intersecting_via_perturb(
                em_sv, d)
            # print("novel_idxs_of_adj_edges", novel_idxs_of_adj_edges)

            # New vertices
            lv = len(vs)
            vs = torch.vstack([vs, new_vertices])
            v_sv = torch.vstack([
                torch.hstack([v_sv, old_vert_new_sv]),
                torch.hstack(
                    [em_sv, torch.zeros(len(new_vertices), 1, dtype=torch.int8, device=device)]),
            ])
            del new_vertices, em_sv, old_vert_new_sv
            # new edges
            # Connect one old to one new index
            new_edges = torch.stack([
                edges[splitting_edge_mask].T.flatten(),
                (lv + torch.arange(splitting_edge_mask.sum(), device=device)).repeat(2)]).T
            edges = torch.vstack([
                # non-splitting edges: take old
                edges[~splitting_edge_mask],
                # split edges: connect one old and one new vertex. Since the new vertices are added in order to vs, just increment index.
                new_edges,
                # intersecting edges
                novel_idxs_of_adj_edges + lv,

            ])
            del splitting_edge_mask, novel_idxs_of_adj_edges
            # print("v_sv finish", v_sv)

        if return_intermediate:
            intermediates[l] = (vs.clone(), edges.clone(), v_sv.clone())

    if return_intermediate:
        return intermediates
    else:
        return vs, edges, v_sv
