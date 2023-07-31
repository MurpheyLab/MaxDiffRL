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
from termcolor import cprint
import numpy as np

from .entropy import get_entropy_params
from utils import _batch_mv

use_jit = True

class MaxDiff(object):

    def __init__(self, model, samples=10, horizon=10, lam=0.1,
                 eps=0.3,barrier=None,bound=1e10,device='cpu',
                 alpha=0.2,explr_dim=None,window=True,weights=None,angle_idx=None,
                 use_real_env=False,cost_to_go=False,control_variance_term=False,
                 receding=False,gamma=1.0, tensor=False):

        self.device          = device
        self.tensor          = tensor

        self.model           = model
        self.num_states      = model.num_states         # additions to PathIntegral
        self.num_actions     = model.num_actions
        if tensor:
            self.num_states  = self.num_states.item()   # additions to PathIntegral
            self.num_actions = self.num_actions.item()
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
        self.eps             = Normal(torch.zeros(self.samples, self.num_actions,device=self.device),
                                        torch.ones(self.samples, self.num_actions,device=self.device) * eps)
        self.explr_dim       = explr_dim                # additions to PathIntegral
        self.angle_idx       = angle_idx                # additions to PathIntegral
        self.weights         = weights                  # additions to PathIntegral
        self.window          = window                   # additions to PathIntegral
        self.alpha           = alpha                    # additions to PathIntegral
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
        self.set_entropy_params(print_params=False)

    def set_entropy_params(self,logdet_method='abs',print_params=False):
        # additions to PathIntegral
        entropy_params       = get_entropy_params(self.t_H,self.num_states,self.device,self.explr_dim,self.angle_idx,self.weights,self.window,logdet_method)
        if print_params: cprint(entropy_params,'cyan')
        if use_jit:
            self.get_entropy     = torch.jit.trace(entropy_params,torch.randn(self.t_H,self.samples,self.num_states,device=self.device),check_trace=False)
            self.get_entropy(torch.randn(self.t_H,self.samples,self.num_states,device=self.device))# prime jit
        else:
            self.get_entropy = entropy_params

    def update_horizon(self,horizon):
        self.t_H = horizon
        self.set_horizon_params(copy_a=True)

    def update_alpha(self,alpha,logdet_method=None):
        self.alpha = alpha
        if logdet_method is not None:
            self.set_entropy_params(logdet_method=logdet_method)
        else:
            self.set_entropy_params()

    def __call__(self, state, eval=False, eval_ctrl=False):

        with torch.no_grad():
            # start = time.time()
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()
            sk              = torch.zeros(self.t_H,self.samples,device=self.device)
            barrier_cost    = torch.zeros(self.t_H,self.samples,device=self.device)
            log_prob        = torch.zeros(self.t_H,self.samples,device=self.device)
            da              = torch.zeros(self.t_H,self.samples,self.num_actions,device=self.device)
            eta             = torch.zeros(self.samples, self.num_actions,device=self.device)
            dones           = torch.zeros(self.t_H,self.samples,device=self.device)
            states          = torch.zeros(self.t_H,self.samples,self.num_states,device=self.device)


            if self.use_real_env:
                s = state[None,:].repeat(self.samples,0)
                self.model.set_state(s)
                sk = []
                states = []
            else:
                if not self.tensor:
                    s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    s = s0.repeat(self.samples, 1)
                else:
                    s = state.repeat(self.samples, 1)

            for t in range(self.t_H):
                if self.use_real_env:
                    states.append(s.copy())
                else:
                    states[t] = s.detach().clone()
                eps = self.eps.sample()
                eta = 0.5 * eta + (1-0.5) * eps
                log_prob[t] = self.eps.log_prob(eta).sum(1)
                da[t] = eta.detach().clone()
                v = self.a[t].expand_as(eta) + eta
                if self.use_real_env:
                    s, rew, _, _ = self.model.step(v.cpu().clone().numpy())
                    barrier_cost[t] = self.barrier.cost(torch.FloatTensor(s).to(self.device))
                    sk.append(rew.copy())
                else:
                    # s, rew = self.model.step(s, v) # doesnt work for traced model
                    s, _, rew, done = self.model(s, v)
                    # error handling
                    s = torch.clamp(s,-self.bound,self.bound) # prevent crazy unbounded stuff from happening
                    barrier_cost[t] = self.barrier.cost(s.detach().clone())
                    sk[t] = rew.squeeze()

            if self.use_real_env:
                sk = torch.FloatTensor(np.stack(rews)).to(self.device)
                states = torch.FloatTensor(np.stack(states)).to(self.device)

            if self.receding:
                sk = sk*self.gammas

            # if self.ctrl_var_explr and not eval:
            if self.ctrl_var_explr or eval_ctrl:
                sk = sk + self.lam*log_prob

            if self.cost_to_go:
                sk = torch.cumsum(sk.flip(0), 0).flip(0)
                if not eval:
                    # S = self.get_entropy(states)
                    # sk = sk + self.alpha*S.repeat(self.t_H,1)
                    S = torch.zeros(self.t_H,self.samples,device=self.device)
                    last_ent_step = self.t_H-len(self.explr_dim)*2
                    for idx in range(last_ent_step):
                        tmp_state = states[idx:,:,:]
                        S[idx] = self.get_entropy(tmp_state)
                    S_avg = torch.mean(S[:last_ent_step,:],dim=0)
                    S[last_ent_step:] = S_avg
                    sk = sk + self.alpha*S
                sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
                sk = sk - barrier_cost

                w = torch.exp(sk.div(self.lam)) + 1e-5
                w.div_(torch.sum(w, dim=1, keepdim=True))
                self.a = self.a + _batch_mv(torch.transpose(da,-1,-2), w)
            else:
                sk = torch.sum(sk,0)
                if not eval:
                    S = self.get_entropy(states)
                    sk = sk + self.alpha*S
                sk = sk - torch.max(sk)
                sk = sk - torch.sum(barrier_cost,0)
                w = torch.exp(sk.div(self.lam)) + 1e-5
                w.div_(torch.sum(w))
                self.a = self.a + torch.transpose(da,-1,-2) @ w
            # print(time.time()-start)
            if self.tensor:
                return self.a[0].detach().clone()
            else:
                return self.a[0].cpu().clone().numpy()
