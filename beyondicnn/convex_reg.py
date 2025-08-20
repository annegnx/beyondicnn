import torch

from beyondicnn.edges_subdivision import skeletal_subdivision, get_e_sv
from beyondicnn.check_convexity import get_convexity_constraints, check_convexity


def convex_regularisation(model, bbox=(-10, 10), device="cpu"):
    vs, edges, v_sv = skeletal_subdivision(
        model, device=device,  bbox=bbox, plot=True, return_intermediate=False)
    e_sv = get_e_sv(v_sv, edges)
    constraints = get_convexity_constraints(model, e_sv)
    constraints = torch.cat(constraints, dim=0)
    # print(constraints)
    # min_constraints = torch.min(torch.FloatTensor(constraints))
    sum_constraints = constraints[constraints < 0].sum()
    return torch.nn.functional.relu(-sum_constraints), constraints
