#!/usr/bin/env python3

import numpy as np
from termcolor import cprint

class PointMass(object):
    __attributes__={'beta','start_distance','boundary','dt','start_mode','explr_dim','noise','manual_seed','integration_method','done_distance'}
    def __init__(self,beta,A,B,start_distance,boundary_distance,dt,explr_dim,noise,start_mode,integration_method,done_distance,clip_rew=False):
        self.beta = beta
        self.manual_seed = None
        self.vel_mul = 0. 
        self.loc_mul = 1.0
        self.transition_noise = False

        self.dt = dt
        self._A = A
        self._B = B
        self.num_states,self.num_actions = B.shape
        self.state = np.zeros(self.num_states)
        self.explr_dim = explr_dim
        self.integration_method = integration_method
        self.done_distance = done_distance
        self.clip_rew = clip_rew
        if clip_rew:
            self.clip = np.square(start_distance*10.)*len(self.explr_dim)

        self.boundary = boundary_distance
        self.noise = noise
        self.start_mode = start_mode
        self.start_pose_idx = 0
        if start_mode == 'random':
            self.start_distance = np.sqrt(np.square(start_distance)*len(explr_dim))
        elif self.start_mode == 'circle10':
            num_pts = 10
            self.start_distance = np.sqrt(np.square(start_distance)*len(explr_dim))
            offset = np.arccos(1/self.start_distance) + np.pi
            angle_increment = np.pi*2./num_pts
            self.base_set = [[self.start_distance*np.cos(angle_increment*idx+offset),
                             self.start_distance*np.sin(angle_increment*idx+offset)] for idx in range(num_pts)
                             ]
        elif self.start_mode == 'four_corners':
            self.base_set = [[-1, -1],[1,-1],[1,1],[-1,1]]
            self.start_distance = start_distance
        else:
            self.start_distance = start_distance

        cprint(['A',self._A,'B',self._B],'red')

    def seed(self,seed):
        self.npr = np.random.RandomState(seed)
        self.manual_seed = seed

    def reset(self):
        self.state = np.zeros(self.num_states)
        if self.start_mode == 'one_corner':
            ##### lower left from origin #####
            self.state[0] = - self.start_distance
            self.state[1] = - self.start_distance
        elif self.start_mode == 'four_corners':
            ##### one of four poses around origin (4 corners) #####
            signs = self.base_set[self.start_pose_idx]
            self.start_pose_idx = (self.start_pose_idx + 1) % 4
            self.state[0] = self.start_distance*signs[0]
            self.state[1] = self.start_distance*signs[1]
        elif self.start_mode == 'random':
            ##### random pose fixed distance from origin #####
            explr_state = self.npr.uniform(-1.,1.,size=len(self.explr_dim))
            explr_state = explr_state / np.linalg.norm(explr_state)*self.start_distance
            self.state[self.explr_dim] = explr_state
        elif self.start_mode == 'circle10':
            pose = self.base_set[self.start_pose_idx]
            self.start_pose_idx = (self.start_pose_idx + 1) % 10
            self.state[self.explr_dim] = pose
        else:
            raise ValueError('invalid start mode requested')
        if self.noise:
            noise = self.npr.normal(scale=0.01,size=self.num_states)
            self.state += noise
        return self.state.copy()

    def f(self,x,u): # transition model
        if self.transition_noise:
            noise = self.npr.normal(scale=0.1,size=self.num_actions)
            u += noise
        return self._A @ x + self._B @ u

    def integrate(self,xt, ut):
        # rk4
        k1 = self.dt * self.f(xt,ut)
        k2 = self.dt * self.f(xt+k1/2.,ut)
        k3 = self.dt * self.f(xt+k2/2.,ut)
        k4 = self.dt * self.f(xt+k3,ut)
        new_xt = xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
        return new_xt

    def reward(self,state): # reward based on pose
        rew = - np.sum(np.square(state[self.explr_dim]))
        if self.clip_rew:
            return np.clip(rew,-self.clip,self.clip)
        else:
            return rew

    def vel_bonus(self,last_state,current_state,max_vel=10.):
        u = -last_state[self.explr_dim]       # desired vec
        v = (current_state[self.explr_dim] - last_state[self.explr_dim])/self.dt # current vec
        rew = np.sum(np.clip(np.sign(u)*v,-max_vel,max_vel))
        return rew-max_vel*len(self.explr_dim) # shift max to 0

    def step(self,u):
        if self.integration_method == 'rk4':
            observation = self.integrate(self.state, u)
        else :
            observation = self.state + self.f(self.state,u) * self.dt

        loc_reward = self.reward(observation.copy())
        vel_reward = self.vel_bonus(self.state.copy(),observation.copy())
        reward = self.loc_mul*loc_reward + self.vel_mul*vel_reward

        self.state = observation.copy()
        if self.boundary > 0:
            at_boundary = np.any(np.abs(observation[self.explr_dim]) > self.boundary)
        else:
            at_boundary = False
        if self.done_distance > 0:
            at_goal = np.linalg.norm(observation[self.explr_dim]) < self.done_distance
        else:
            at_goal = False
        done = at_boundary or at_goal
        return observation, reward, done, {}

    def close(self):
        pass

    def __repr__(self):
        out = [x+': '+str(self.__dict__[x]) for x in self.__attributes__]
        return self.__name__ + '( \n  ' + ', \n  '.join(out) + '\n )'

class PointMass2D_DoubleIntEnv(PointMass):
    __name__ = 'PointMass2D_DoubleIntEnv'
    def __init__(self,beta=1.,gamma=1.,start_distance=1.0,boundary_distance=10.,dt=0.1,noise=False,start_mode='one_corner',integration_method='rk4',done_distance=0.01):
        # x = [x y xdot ydot]^T
        A = np.array([[0.,0.,beta,0.],
                      [0.,0.,0.,gamma],
                      [0.,0.,0.,0.],
                      [0.,0.,0.,0.]])
        B = np.array([[0.,0.],
                      [0.,0.],
                      [1.,0.],
                      [0.,1.]])
        explr_dim = [0,1]
        super().__init__(beta,A=A, B=B, dt=dt, start_distance=start_distance, boundary_distance=boundary_distance,
                            explr_dim=explr_dim,noise=noise, start_mode=start_mode, integration_method=integration_method,done_distance=done_distance)

class PointMass2D_TripleIntEnv(PointMass):
    __name__ = 'PointMass2D_TripleIntEnv'
    def __init__(self,beta=1.,gamma=1.,start_distance=1.0,boundary_distance=10.,dt=0.1,noise=False,start_mode='one_corner',integration_method='rk4',done_distance=0.01):
        # x = [x y xdot ydot xddot yddot]^T
        A = np.array([[0.,0.,beta,0.,0.,0.],
                      [0.,0.,0.,gamma,0.,0.],
                      [0.,0.,0.,0.,beta,0.],
                      [0.,0.,0.,0.,0.,gamma],
                      [0.,0.,0.,0.,0.,0.],
                      [0.,0.,0.,0.,0.,0.]])
        B = np.array([[0.,0.],
                      [0.,0.],
                      [0.,0.],
                      [0.,0.],
                      [1.,0.],
                      [0.,1.]])
        explr_dim = [0,1]
        super().__init__(beta,A=A, B=B, dt=dt, start_distance=start_distance, boundary_distance=boundary_distance,
                            explr_dim=explr_dim,noise=noise, start_mode=start_mode, integration_method=integration_method,done_distance=done_distance)
