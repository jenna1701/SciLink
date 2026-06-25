"""Changeover-aware Expected Improvement — a skill-shipped acquisition (issue
#196). A genuinely BEYOND-BoTorch, lab-operational acquisition.

Real labs pay a large setpoint-CHANGEOVER cost between consecutive experiments:
re-stabilizing a furnace to a new temperature can take hours; switching solvent
or substrate means cleaning and re-priming. Standard BO ignores this — it jumps
to the global EI optimum no matter how far that is from the current setpoint, so
the campaign wastes time on changeovers. (BoTorch's cost-aware utilities model
multi-fidelity EVALUATION cost, not move/changeover cost between sequential
experiments — there is no such acquisition in BoTorch.)

`changeover_ei` discounts Expected Improvement by how far a candidate moves the
*expensive-to-change* inputs from the last experiment's setpoint, trading a
little improvement for much less changeover overhead.

Params (set by the agent from the skill guidance):
  - ``expensive_dims``      : indices of slow/expensive-to-change inputs
                              (e.g. the temperature column). Default: all inputs.
  - ``changeover_weight``   : λ ≥ 0; higher = penalize moves harder. Default 1.0.
"""
import numpy as np

from scilink.skills._shared._opt_components import AcquisitionComponent


def _recommend_changeover_ei(optimizer, n_candidates, params):
    import torch
    from botorch.acquisition import ExpectedImprovement
    from botorch.utils.sampling import draw_sobol_samples

    p = params or {}
    lam = float(p.get("changeover_weight", 1.0))
    d = optimizer.input_dim
    dims = list(p.get("expensive_dims") or range(d))
    bounds = optimizer.bounds
    span = (bounds[1] - bounds[0]).clamp_min(1e-9)
    last = optimizer.X_train[-1]  # most recent experiment = current setpoint

    pool = draw_sobol_samples(bounds=bounds, n=4096, q=1).squeeze(1)  # (N, d)
    acq = ExpectedImprovement(model=optimizer.model, best_f=optimizer.y_train.max())
    with torch.no_grad():
        ei = acq(pool.unsqueeze(1)).clamp_min(0.0)                              # (N,)
        move = ((pool[:, dims] - last[dims]).abs() / span[dims]).sum(dim=-1)    # (N,) normalized
        score = ei / (1.0 + lam * move)                                        # EI per unit changeover
    top = torch.topk(score, n_candidates).indices
    return pool[top].detach().cpu().numpy()


ACQUISITION_SPEC = AcquisitionComponent(
    key="changeover_ei",
    recommend_fn=_recommend_changeover_ei,
    agents=["bo"],
    description=("Changeover-aware Expected Improvement: discounts EI by how far a "
                "candidate moves expensive-to-change setpoints from the last "
                "experiment. Params: expensive_dims (list), changeover_weight (float)."),
)
