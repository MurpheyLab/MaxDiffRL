#!/usr/bin/env python3
import gym
import numpy as np
import os
import mujoco_py
from scipy.spatial.transform import Rotation

def angle_wrap(angles):
    # -pi to pi
    return (angles + np.pi) % (2 * np.pi) - np.pi

'''
Half Cheetah
'''
class HalfCheetahRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._healthy_theta_boundary = np.pi/6
        self._unhealthy_theta_range  = np.pi - self._healthy_theta_boundary

    def reward(self, rew):
        # angle penalty
        theta = np.abs(angle_wrap(self.sim.data.qpos[2])) # 0 to pi
        if theta > self._healthy_theta_boundary:
            err = np.square( (self._healthy_theta_boundary - theta) / self._unhealthy_theta_range )
        else:
            err = 0.
        return rew-0.1*err

'''
Ant
'''

class AntBasicContactsWrapper(gym.Wrapper):
    def __init__(self, env, include_contacts=True, **kwargs):
        super().__init__(env)
        self.include_contacts = include_contacts
        if not(self.include_contacts):
            obs = self._get_obs()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(len(obs),))

    def _update_contacts(self):
        # with mujoco200 + mujoco_py >= 2.0.0, need to calculate contacts
        mujoco_py.functions.mj_rnePostConstraint(self.env.sim.model, self.env.sim.data)

    def step(self, action):
        # 1. run general step
        _, reward, done, info = self.env.step(action) # reward = (xposafter - xposbefore) / self.dt
        # 2. update contacts and reward
        if self.include_contacts:
            self._update_contacts()
            # replace old contact cost with new contact cost
            old_rew_contact = info['reward_contact']
            new_rew_contact = -self.env.contact_cost
            reward = reward - old_rew_contact + new_rew_contact
            info['reward_contact'] = new_rew_contact
        # 3. fix observation to account for contacts (or lack of contacts)
        new_obs = self._get_obs()
        return new_obs, reward, done, info

    def _get_obs(self):
        position = self.env.sim.data.qpos.flat.copy()
        velocity = self.env.sim.data.qvel.flat.copy()
        if self.include_contacts:
            contact_force = self.env.contact_forces.flat.copy()
            observations = np.concatenate((position, velocity, contact_force))
        else:
            observations = np.concatenate((position, velocity))
        return observations

    def reset(self):
        self._upside_down_start_time = None
        state = self.env.reset()
        if self.include_contacts:
            self._update_contacts()
        new_obs = self._get_obs()
        return new_obs

from gym.envs.robotics import rotations

class AntContactsWrapper(gym.Wrapper):
    def __init__(self, env, task, desired_speed=1.0, include_contacts=True,mod_done=True,no_done=False):
        super().__init__(env)
        self.env = env
        self.task = task
        self.include_contacts = include_contacts
        if not(self.include_contacts):
            obs = self._get_obs()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(len(obs),))
        # upside down if both boundaries are violated
        self._upright_theta= 0.35
        self._unhealthy_theta_range = np.pi - self._upright_theta
        self._healthy_theta_boundary = 2.7
        # self._healthy_z_boundary = 0.3
        self._upside_down_start_time = None
        self._stuck_thresh = 1.
        self._desired_speed = desired_speed
        self.mod_done = mod_done
        self.no_done = no_done

    def _update_contacts(self):
        # with mujoco200 + mujoco_py >= 2.0.0, need to calculate contacts
        mujoco_py.functions.mj_rnePostConstraint(self.env.sim.model, self.env.sim.data)

    def _done(self,state):
        quat = state[3:7]
        zz = rotations.quat2mat(quat)[2,2]
        theta = np.arccos(zz)
        z = state[2]

        # check if agent is stuck upside down
        is_upside_down =  (theta > self._healthy_theta_boundary) # (z < self._healthy_z_boundary) and
        duration = 0.
        if is_upside_down:
            current_time = self.env.sim.data.time
            if self._upside_down_start_time is None:
                self._upside_down_start_time = current_time
            else:
                duration = current_time - self._upside_down_start_time
        else:
            self._upside_down_start_time = None
        is_healthy = np.isfinite(state).all() and (duration < self._stuck_thresh)
        return not is_healthy

    def _upright(self,state):
        # 0 = upright, -1 = upside down, decay between _upright_theta and pi
        quat = state[3:7]
        zz = rotations.quat2mat(quat)[2,2]
        theta = np.arccos(zz)

        if theta > self._upright_theta:
            err = np.square( (self._upright_theta - theta) / self._unhealthy_theta_range )
        else:
            err = 0.
        return -err

    def err_run_x(self,forward_vel,eps=0.1):
        if np.abs(forward_vel-self._desired_speed) > eps:
            err = np.square(forward_vel-self._desired_speed)
        else:
            err = 0.
        return - err

    def err_run_x_narrow(self,forward_vel):
        return - np.square(forward_vel-self._desired_speed)*20

    def step(self, action):
        # 1. run general step
        _, reward, done, info = self.env.step(action) # reward = (xposafter - xposbefore) / self.dt
        # 2. upate contacts
        self._update_contacts()
        # 3. fix observation to account for contacts (or lack of contacts)
        new_obs = self._get_obs()
        # 4. update reward
        ## if you want to include contact cost uncomment next 2 lines
            # contact_cost = self.env.contact_cost
            # reward = reward - contact_cost
        if self.task == 'run_x':
            reward = self.err_run_x(reward)
        elif self.task == 'run_x2':
            reward = reward + self.err_run_x(reward)
        elif self.task == 'run_x3':
            reward = reward + 0.25*self.err_run_x(reward)
        elif self.task == 'run_x4':
            reward = 2. + self.err_run_x(reward)
        elif self.task == 'RunX':
            ''' offset by 6 because max reward for mujoco gym ant is 6000 '''
            reward = 6. + self.err_run_x(reward)
        elif self.task == 'upright':
            reward = reward + 0.1*self._upright(new_obs)
        elif self.task == 'orig':
            pass
        # 5. see if done
        if self.no_done:
            done = False
        elif self.mod_done:
            done = self._done(new_obs)

        return new_obs, reward, done, info

    def _get_obs(self):
        position = self.env.sim.data.qpos.flat.copy()
        velocity = self.env.sim.data.qvel.flat.copy()
        if self.include_contacts:
            contact_force = self.env.contact_forces.flat.copy()
            observations = np.concatenate((position, velocity, contact_force))
        else:
            observations = np.concatenate((position, velocity))
        return observations

    def reset(self):
        self._upside_down_start_time = None
        state = self.env.reset()
        self._update_contacts()
        new_obs = self._get_obs()
        return new_obs


# class AntNoContactsWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         obs = self._get_obs()
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(len(obs),))
#
#     def reset(self):
#         state = self.env.reset()
#         # remove contacts from observation
#         new_obs = self._get_obs()
#         return new_obs
#
#     def step(self, action):
#         next_state, reward, done, _ = self.env.step(action)
#         # remove contacts from observation
#         new_obs = self._get_obs()
#         return new_obs, reward, done, _
#
#     def _get_obs(self):
#         position = self.env.sim.data.qpos.flat.copy()
#         velocity = self.env.sim.data.qvel.flat.copy()
#         observations = np.concatenate((position, velocity))
#         return observations


'''
Linearize angle (agent in world frame)
'''

class Linearize(gym.Wrapper):
    _angle_locs = {
        'SwimmerEnv' : 0,
        'SwimmerEnv_v3' : 2,
        'HopperEnv' : 1,
        'HopperEnv_v3' : 2,
        'HalfCheetahEnv' : 1,
        'HalfCheetahEnv_v3' : 2,
    }
    def __init__(self, env, env_name):
        if env_name not in self._angle_locs.keys():
            raise ValueError('env linearization not implemented')
        super().__init__(env)
        self.angle_loc = self._angle_locs[env_name]

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self._linearize_angle(next_state), reward, done, info

    def reset(self):
        state = self.env.reset()
        return self._linearize_angle(state)

    def _linearize_angle(self,state):
        angle = state[self.angle_loc]
        linearized = np.array([np.sin(angle),np.cos(angle)])
        obs = np.concatenate((state[:self.angle_loc],linearized, state[1+self.angle_loc:]))
        return obs
