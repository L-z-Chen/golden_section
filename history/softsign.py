
import torch
from torch.optim.optimizer import Optimizer
def exists(val):
  return val is not None

# update functions

def softsign_update_fn(p, grad, momentum, lr, wd, beta, eps, correct_bias, step):
  # apply decoupled weight decay
  if wd != 0:
    p.data.mul_(1. - lr * wd)

  # accumulate momentum
  momentum.mul_(beta).add_(grad, alpha=1. - beta)
  if correct_bias:
    bias_correction = 1.0 - beta ** step
    momentum_hat = momentum / bias_correction
  else:
    momentum_hat = momentum
  
  med_grad = grad.abs().median()
  soft_sign = momentum_hat / (momentum_hat.abs() + eps * med_grad)

  # Replace NaNs with the sign of momentum_hat
  p.add_(soft_sign, alpha=-lr)

class SoftSign(Optimizer):
  def __init__(
    self,
    params,
    lr: list = [-5., -2.], # log scale
    beta: list = [0.6, 0.99], 
    eps: list = [-12.3, -1.], # log scale
    weight_decay: list = [-3., -.7], # log scale
    correct_bias: bool = True,
  ):
    assert lr > 0.
    assert 0. <= beta <= 1.

    defaults = dict( 
      lr = lr,
      beta = beta,
      eps=eps,
      weight_decay = weight_decay,
      correct_bias = correct_bias
    )

    super().__init__(params, defaults)

    self.update_fn = softsign_update_fn

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      for p in filter(lambda p: exists(p.grad), group['params']):
        grad, lr, beta, eps, correct_bias, state = (
          p.grad, group['lr'], group['beta'], group['eps'], group['correct_bias'], self.state[p]
        )

        # initialize state
        if len(state) == 0:
          state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state['step'] = 0

        momentum = state['exp_avg']
        state['step'] += 1

        # update parameters
        self.update_fn(
          p,
          grad,
          momentum,
          lr,
          weight_decay,
          beta,
          eps,
          correct_bias,
          state['step']
        )

    return loss
