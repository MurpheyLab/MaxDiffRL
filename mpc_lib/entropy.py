#!/usr/bin/env python3

import torch
from torch.distributions import Normal
from utils import jit_prob_dist

def cov(tensor: torch.Tensor, rowvar: bool=True, bias:bool=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/Moself.darTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

''' entropy utils '''

class GetEntropy(torch.nn.Module):
    __tensor_attributes__ = (  'explr_dim','cov_weights','angle_idx','quat_specs','mean','std')
    __attributes__ = ( 'full_dim','window_size','step_size','num_windows','convert_quat')
    __logdet_methods__ = ( 'abs','mean','min','max')
    def __init__(self,full_dim,explr_dim,cov_weights,angle_idx,convert_quat,quat_specs,window_size,step_size,num_windows,explr_size,device,logdet_method,weight_method):
        super(GetEntropy,self).__init__()
        self.full_dim = full_dim
        self.register_buffer('explr_dim',explr_dim)
        self.register_buffer('cov_weights',cov_weights)
        self.register_buffer('angle_idx',angle_idx)
        self.convert_quat = convert_quat
        self.register_buffer('quat_specs',quat_specs)
        self.window_size = window_size
        self.step_size = step_size
        self.num_windows = num_windows
        self.explr_size = explr_size
        mean = torch.zeros(explr_size,explr_size,device=device)
        std = torch.ones(explr_size,explr_size,device=device)*(1e-8)
        self.register_buffer('mean',mean)
        self.register_buffer('std',std)
        self.noise = jit_prob_dist(Normal(mean,std))
        assert logdet_method in self.__logdet_methods__, f'requested logdet_method not implement, choose from {self.__logdet_methods__}'
        self.logdet_method = logdet_method
        self.weight_method = weight_method

    def extra_repr(self):
        out = []
        out = out + [x+': '+str(self._buffers[x].data) for x in self.__tensor_attributes__]
        out = out + [x+': '+str(self.__dict__[x]) for x in self.__attributes__]
        return ', \n'.join(out)

    def forward(self,tensor:torch.Tensor):
        if not self.full_dim:
            tensor = tensor[:,:,self.explr_dim]  # select explr dims
        traj = torch.transpose(tensor,0,1) # reorganize to make dims [traj#, tstep, state]
        if self.window_size == 0:
            traj = torch.transpose(traj.unsqueeze(1),-1,-2) # dim, size, step
        else:
            traj = traj.unfold(1,self.window_size,self.step_size) # dim, size, step
        if 'quad' in self.weight_method:
            C = self.cov_weights @ cov(traj) @ self.cov_weights
        elif 'mul' in self.weight_method:
            C = cov(traj)*self.cov_weights
        # if noise_list is None:
        noise_list = self.noise.sample([traj.shape[0],self.num_windows]) # numerical stability
        C = C + noise_list

        if self.logdet_method == 'abs':
            sign,logdet = torch.slogdet(C)
        else:
            logdet = torch.logdet(C)
            nans = logdet.isnan()
            if self.logdet_method == 'mean':
                nan_logdet = torch.mean(logdet[~nans]) # sanitycheck0
            elif self.logdet_method == 'min':
                nan_logdet = torch.min(logdet[~nans]) # sanitycheck1
            elif self.logdet_method == 'max':
                nan_logdet = torch.max(logdet[~nans]) # sanitycheck2
            logdet[nans] = nan_logdet

        S = torch.mean(0.5*logdet,dim=1)
        return S


def get_entropy_params(horizon,num_states,device,explr_dim,angle_idx,weights,window,logdet_method='abs',weight_method='quad'):
    convert_quat = False
    quat_specs = []
    if explr_dim is None or (explr_dim=='None'):
        explr_dim = []
        full_dim = True
        explr_size = num_states
    else:
        full_dim = False
        explr_size = len(explr_dim)
    if angle_idx is None or (angle_idx=='None'):
        angle_idx = []
    if weights is None or (weights=='None'):
        weights = [1.]*explr_size
    cov_weights = torch.diag(torch.FloatTensor(weights))
    if window:
        window_size = max(int(horizon/4),explr_size*2)
        step_size = int(window_size/3)
        num_windows = torch.zeros(horizon).unfold(0,window_size,step_size).shape[0]
    else:
        window_size = 0
        step_size = 1
        num_windows = 1

    explr_dim = torch.tensor(explr_dim)
    angle_idx = torch.tensor(angle_idx)
    quat_specs = torch.tensor(quat_specs)
    return GetEntropy(full_dim, explr_dim,cov_weights,angle_idx,convert_quat,quat_specs,
                        window_size,step_size,num_windows,explr_size,device,logdet_method,weight_method).to(device)
