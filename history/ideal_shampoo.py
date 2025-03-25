import math
import torch

@torch.compile
def _matrix_power(matrix: torch.Tensor) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(-0.5).diag() @ v.t()).to(device)


class Dove(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        ns_steps=5,
        matrix_beta1 = 0.9,
        matrix_beta2 = 0.95,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        alpha = 1.0,
        embed_dim_threshold=2000,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            ns_steps=ns_steps,
            matrix_betas=(matrix_beta1, matrix_beta2),
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            alpha = alpha
        )
        muon_params = []
        adamw_params = []

        for p in params:
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and max(p.shape) <= embed_dim_threshold:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        all_params = muon_params + adamw_params
        super().__init__(all_params, defaults)

        for p in muon_params:
            assert p.ndim == 2
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            alpha = group["alpha"]
            norms = {}
            # generate weight updates in distributed fashion
            for i, p in enumerate(params):
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "matrix_step" not in state:
                    state["matrix_step"] = 0
                    state["L"] = torch.zeros(g.size(0), g.size(0), device=g.device)
                    state["R"] = torch.zeros(g.size(1), g.size(1), device=g.device)
                    state["M"] = torch.zeros_like(g)
                
                M = state["M"]
                L = state["L"]
                R = state["R"]
                state["matrix_step"] += 1
                step = state["matrix_step"]
                beta1, beta2 = group["matrix_betas"]
                
                M.mul_(beta2).add_(g, alpha=1 - beta2)
                L.mul_(beta2).add_(M @ M.t(), alpha=1 - beta2)
                R.mul_(beta2).add_(M.t() @ M, alpha=1 - beta2)
                # bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                inv_L = _matrix_power(L/bias_correction2
                                       + 1e-8 * torch.eye(L.size(0), device=L.device))
                inv_R = _matrix_power(R/bias_correction2
                                       + 1e-8 * torch.eye(R.size(0), device=R.device))
                
                update = inv_L @ (M/bias_correction1) @ inv_R * torch.trace(L/bias_correction2)**0.5
                norms[f"{i}_{tuple(p.shape)}"] = (lr * wd * p + update * lr * alpha).norm()/ p.numel()**0.5
                # apply weight decay
                p.data.mul_(1 - lr * wd)
                # apply update
                p.data.add_(update, alpha=-lr * alpha)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                norms[f"{i}_{tuple(p.shape)}"] = (lr * wd * p + g*lr/scale).norm()/ p.numel()**0.5

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss, norms
