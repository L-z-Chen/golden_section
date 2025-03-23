import math
import torch


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: list = [-5. , -2.],  # log scale
        wd: list = [-3., -0.7],  # log scale
        momentum: float = 0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        embed_dim_threshold=1000,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
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

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW backup
            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in adamw_params:
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

                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr / scale)

        return loss
