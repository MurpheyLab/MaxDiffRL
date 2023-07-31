#!/usr/bin/env python3
'''
Williams, G., Wagener, N., Goldfain, B., Drews, P., Rehg, J.M., Boots, B. and
Theodorou, E.A., 2017, May. Information theoretic MPC for model-based
reinforcement learning. In 2017 IEEE International Conference on Robotics
and Automation (ICRA) (pp. 1714-1721). IEEE.
'''
import torch
from torch.distributions import Normal
import time
import numpy as np
from utils import _batch_mv

class PathIntegral(object):

    def __init__(self, model, samples=10, horizon=10, lam=0.1,
                 eps=0.3, barrier=None, bound=1e10, device='cpu',
                 use_real_env=False,cost_to_go=False,control_variance_term=True,
                 receding=False,gamma=1.0,tensor=False):

        self.device          = device
        self.tensor          = tensor

        self.model           = model
        self.num_actions     = model.num_actions
        if tensor:
            self.num_actions = int(self.num_actions.item())
        self.t_H             = horizon
        self.lam             = lam
        self.samples         = samples
        self.bound           = bound
        if barrier is not None:
            self.barrier     = barrier
        else:
            from .utils import EmptyBarrier
            self.barrier     = EmptyBarrier()
        self.gamma           = gamma
        self.eps             = Normal(torch.zeros(self.samples, self.num_actions).to(self.device),
                                    torch.ones(self.samples, self.num_actions).to(self.device) * eps)
        self.set_horizon_params(print_params=True)

        # flags
        self.use_real_env    = use_real_env
        self.cost_to_go      = cost_to_go
        self.receding        = receding
        self.ctrl_var_explr  = control_variance_term


    def reset(self):
        with torch.no_grad():
            self.a.zero_()

    def set_horizon_params(self,print_params=False,copy_a=False):
        gammas               = self.gamma**torch.arange(self.t_H,device=self.device)
        self.gammas          = gammas.unsqueeze(-1).repeat(1,self.samples)
        if copy_a:
            old_a = self.a.clone()
        self.a               = torch.zeros(self.t_H, self.num_actions,device=self.device)
        if copy_a:
            self.a[:old_a.shape[0]] = old_a

    def update_horizon(self,horizon):
        self.t_H = horizon
        self.set_horizon_params(copy_a=True)

    def __call__(self,state,eval=False, eval_ctrl=False):

        with torch.no_grad():
            start = time.time()
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()
            sk              = torch.zeros(self.t_H,self.samples,device=self.device)
            barrier_cost    = torch.zeros(self.t_H,self.samples,device=self.device)
            log_prob        = torch.zeros(self.t_H,self.samples,device=self.device)
            da              = torch.zeros(self.t_H,self.samples,self.num_actions,device=self.device)
            eta             = torch.zeros(self.samples, self.num_actions,device=self.device)
            dones           = torch.zeros(self.t_H,self.samples,device=self.device)

            if self.use_real_env:
                self.model.set_state(state[None,:].repeat(self.samples,0))
                sk = []
            else:
                if not self.tensor:
                    s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    s = s0.repeat(self.samples, 1)
                else:
                    s = state.repeat(self.samples, 1)

            for t in range(self.t_H):
                eps = self.eps.sample()
                eta = 0.5 * eta + (1-0.5) * eps
                log_prob[t] = self.eps.log_prob(eta).sum(1)
                da[t] = eta.detach().clone()
                v = self.a[t].expand_as(eta) + eta
                if self.use_real_env:
                    s, rew, _, _ = self.model.step(v.cpu().clone().numpy())
                    barrier_cost.append(self.barrier.cost(torch.FloatTensor(s).to(self.device)))
                    sk.append(rew.copy())
                else:
                    # s, rew = self.model.step(s, v) # doesnt work for traced model
                    s, _, rew, _ = self.model(s, v)
                    s = torch.clamp(s,-self.bound,self.bound) # prevent crazy unbounded stuff from happening
                    barrier_cost[t] = self.barrier.cost(s.detach().clone())
                    sk[t] = rew.squeeze()

            if self.use_real_env:
                sk = torch.FloatTensor(np.stack(sk)).to(self.device)

            if self.receding:
                sk = sk*self.gammas

            # if self.ctrl_var_explr and not eval:
            if self.ctrl_var_explr or eval_ctrl:
                sk = sk + self.lam*log_prob

            if self.cost_to_go:
                sk = torch.cumsum(sk.flip(0), 0).flip(0)
                sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
                sk = sk - barrier_cost

                w = torch.exp(sk.div(self.lam)) + 1e-5
                w.div_(torch.sum(w, dim=1, keepdim=True))
                self.a = self.a + _batch_mv(torch.transpose(da,-1,-2), w)
            else:
                sk = torch.sum(sk,0)
                sk = sk - torch.max(sk)
                sk = sk - torch.sum(barrier_cost,0)
                w = torch.exp(sk.div(self.lam)) + 1e-5
                w.div_(torch.sum(w))
                self.a = self.a + torch.transpose(da,-1,-2) @ w
            if self.tensor:
                return self.a[0].detach().clone()
            else:
                return self.a[0].cpu().clone().numpy()
