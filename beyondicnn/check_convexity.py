import copy
import torch
import itertools


def copy_and_zero_biases(f, device):
    """Return a copy of the network with biases set to 0"""
    f_copy = copy.deepcopy(f)
    for module in f_copy.modules():
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    return f_copy


def generate_hyperplanes(pre_acts_nu):
    zero_mask = (pre_acts_nu == 0).any(dim=1)

    # Separate rows with and without zeros
    rows_without_zeros = pre_acts_nu[~zero_mask]  # Keep these as is
    rows_with_zeros = pre_acts_nu[zero_mask]

    if rows_with_zeros.shape[0] > 0:
        # print(rows_with_zeros)
        zero_positions = (rows_with_zeros == 0).nonzero(as_tuple=True)

        # Compute number of zeros per row
        num_zeros_per_row = (rows_with_zeros == 0).sum(dim=1)
        # print(num_zeros_per_row)

        # Generate all possible +1/-1 replacements for zero positions

        all_replacement_patterns = [
            torch.tensor(list(itertools.product([-1, 1], repeat=n))) for n in num_zeros_per_row.tolist()
        ]
        # print(all_replacement_patterns)

        # Expand rows based on the number of replacements needed
        expanded_rows = rows_with_zeros.repeat_interleave(2 ** num_zeros_per_row, dim=0)
        # print(expanded_rows.shape)

        # Find the positions where zeros occur in the repeated tensor
        zero_indices = (expanded_rows == 0).nonzero(as_tuple=True)

    # Create a list of expanded replacements
        expanded_replacements = torch.cat([l.view(-1)
                                           for l in all_replacement_patterns], dim=0).to(torch.int8).to(pre_acts_nu.device)

        # Replace zeros with generated values
        expanded_rows[zero_indices] = expanded_replacements.view(-1)

        # Concatenate back with original non-zero rows
        final_pre_acts = torch.cat([rows_without_zeros, expanded_rows], dim=0)

        acts = torch.unique(torch.clip(final_pre_acts, min=0), dim=0)
        return acts
    else:
        return torch.unique(torch.clip(rows_without_zeros, min=0), dim=0)


def check_convexity(model, sign_vertices):

    total_depth = len(model.ks)
    input_dim = model.ks[0]
    d2 = 2*input_dim
    # sign_vertices_rm_corners = sign_vertices[:, 2*input_dim:].clone()
    # zero_counts = (sign_vertices_rm_corners == 0).sum(dim=1)
    # one_skeleton = sign_vertices_rm_corners[zero_counts == input_dim-1]

    model_null_bias = copy_and_zero_biases(model, device=sign_vertices.device)
    condition_per_layer = {}
    is_convex = True
    for l in range(1, total_depth-1):
        K = sum(model.ks[1:l])
        shape_layer = model.ks[l]
        condition_per_neuron = {}
        for k in range(model.ks[l]):
            nu = (sign_vertices[:, d2 + K+k] == 0)
            sign_vertices_nu = sign_vertices[nu, :]
            if l < total_depth-2:
                sign_vertices_nu = torch.unique(
                    sign_vertices_nu[:, d2+K+shape_layer:], dim=0)

                acts = generate_hyperplanes(sign_vertices_nu)
                # acts = torch.clip(sign_vertices[nu, K+shape_layer:], 0)
            else:
                acts = torch.Tensor([1])
            val = torch.zeros((1, shape_layer)).to(sign_vertices.device)
            val[:, k] = 1.0
            subgraph_ks = model.ks[l:]
            subgraph_Ks = [sum(subgraph_ks[1:j]) for j in range(
                1, len(subgraph_ks))] + [sum(subgraph_ks[1:])]  # similar to neuron_idx
            for i, layer in enumerate(model_null_bias.fcs[l:-1]):
                val = layer(val)
                val = acts[:, subgraph_Ks[i]:subgraph_Ks[i+1]] * val
            val = model_null_bias.fcs[-1](val)
            condition_per_neuron[k] = val >= 0
            convex_satisfied = torch.all(val >= 0).item()
            if not convex_satisfied:
                is_convex = False
                with open("training_log_2.txt", "a") as log_file:
                    log_file.write(f"not satified for layer {l}, neuron {k}\n")
                    log_file.write(f"vals {val}\n")
                    log_file.write(f"acts {acts}\n")
                    log_file.write(f"\n")

        condition_per_layer[l] = condition_per_neuron
    return is_convex, condition_per_layer


def get_convexity_constraints(model, sign_vertices):

    total_depth = len(model.ks)
    input_dim = model.ks[0]
    d2 = 2*input_dim
    # sign_vertices_rm_corners = sign_vertices[:, 2*input_dim:].clone()
    # zero_counts = (sign_vertices_rm_corners == 0).sum(dim=1)
    # one_skeleton = sign_vertices_rm_corners[zero_counts == input_dim-1]

    constraints = []
    is_convex = True
    for l in range(1, total_depth-1):
        K = sum(model.ks[1:l])
        # print("layer l ", l)
        shape_layer = model.ks[l]
        constraints_per_neuron = []
        for k in range(model.ks[l]):
            # print("neuron", k)
            # print("K", K, "k", k)
            nu = (sign_vertices[:, d2 + K+k] == 0)
            sign_vertices_nu = sign_vertices[nu, :]
            # print(sign_vertices_nu)
            if l < total_depth-2:
                sign_vertices_nu = torch.unique(
                    sign_vertices_nu[:, d2+K+shape_layer:], dim=0)

                acts = generate_hyperplanes(sign_vertices_nu)
                # print(acts)
                # acts = torch.clip(sign_vertices[nu, K+shape_layer:], 0)
            else:
                acts = torch.Tensor([1])
            val = torch.zeros((1, shape_layer)).to(sign_vertices.device)
            val[:, k] = 1.0
            subgraph_ks = model.ks[l:]
            subgraph_Ks = [sum(subgraph_ks[1:j]) for j in range(
                1, len(subgraph_ks))] + [sum(subgraph_ks[1:])]  # similar to neuron_idx
            for i, layer in enumerate(model.fcs[l:-1]):
                val = val @ layer.weight.T
                val = acts[:, subgraph_Ks[i]:subgraph_Ks[i+1]] * val
            val = val @ model.fcs[-1].weight.T
            constraints_per_neuron.append(val)
            convex_satisfied = torch.all(val >= 0).item()
            if not convex_satisfied:
                is_convex = False
                with open("training_log_2.txt", "a") as log_file:
                    log_file.write(f"not satified for layer {l}, neuron {k}\n")
                    log_file.write(f"vals {val}\n")
                    log_file.write(f"acts {acts}\n")
                    log_file.write(f"\n")

        constraints += constraints_per_neuron
    return constraints
