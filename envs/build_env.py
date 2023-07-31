#!/usr/bin/env python3

__all__=['env_list','getlist','build_env']

''' load environments '''
# environments
from gym.envs import mujoco
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

env_list = {
    'HalfCheetahEnv' : mujoco.HalfCheetahEnv,
    'AntEnv' : mujoco.AntEnv,
    # v3 (with positions)
    'SwimmerEnv_v3' : SwimmerEnv,
    'AntEnv_v3' : AntEnv,
    'HalfCheetahEnv_v3' : HalfCheetahEnv,
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str


''' build environment '''
from .normalized_actions import NormalizedActions
import os

def build_env(args,config,device):
    # initialize environment
    env_name = args.env
    traj = None
    viewer = None
    print('PointMass' in args.env,'PointMass', args.env)
    if 'PointMass' in args.env:
        from envs.pointmass_lib import env_list_pm,TrajBuffer,PM_Viewer
        env = env_list_pm[env_name](beta=args.beta,start_mode=args.start_mode,**config['env'])
        action_dim = env.num_actions
        state_dim  = env.num_states
        # set up plotting
        for name in ['alpha','horizon','env']:
            args.__dict__[name] = 'n/a'
            if name in config.keys():
                args.__dict__[name] = config[name]
            elif 'planner' in config.keys():
                if name in config['planner'].keys():
                    args.__dict__[name] = config['planner'][name]
            elif ('sac' in args.method) and name == 'alpha':
                args.__dict__[name] = config['reward_scale']
        traj = TrajBuffer(args,capacity=11)
        if args.render:
            viewer = PM_Viewer(args)
    else:
        env_args = {}
        if 'xml_file' in args.__dict__.keys():
            env_args['xml_file'] = args.xml
        if not('done_util' in args.__dict__.keys()):
            args.done_util = True
        env_args['render']= args.render
        if args.v3:
            env_args['exclude_current_positions_from_observation']= False
            env_args['terminate_when_unhealthy']= args.done_util
            if env_name=='MultiSwimmer':
                args.env = 'SwimmerEnv_v3'
                env_name = args.env
                multi_swimmer_wrapper = True
            elif env_name=='SwimmerEnv_v3':
                if args.mod_weight.lower()=='light':
                    xml = os.path.abspath( './envs/light_swimmer.xml')
                    env_args['xml_file'] = xml
                    config['name_mod'] = '_Light' + config['name_mod']
                    print('loading light model')
                else:
                    print('loading default model')
            elif env_name=='MultiAnt':
                env_args['ctrl_cost_weight'] = 0.
                env_args['contact_cost_weight'] = 0.
                env_args['healthy_reward'] = 0.
                args.env = 'AntEnv_v3'
                env_name = args.env
                multi_ant_wrapper = True
            elif env_name=='AntEnv_v3':
                # env_args['healthy_z_range'] = (0.26,np.inf) # was set to (0.2,1.0) but didn't actually end when upside down
                env_args['ctrl_cost_weight'] = 0.
                env_args['contact_cost_weight'] = 0.
                env_args['healthy_reward'] = 0.
            elif env_name == 'HalfCheetahEnv_v3':
                pass
            else:
                raise ValueError('invalid env name')
        try:
            env = NormalizedActions(env_list[env_name](**env_args))
        except TypeError as err:
            del env_args['render']
            try:
                # print('no argument render,  assuming env.render will just work')
                env = NormalizedActions(env_list[env_name](**env_args))
            except TypeError as err:
                del env_args['terminate_when_unhealthy']
                # print('no argument terminate_when_unhealthy')
                env = NormalizedActions(env_list[env_name](**env_args))

        if env_name=='AntEnv_v3':
            from envs.wrappers import AntContactsWrapper
            env = AntContactsWrapper(env,**config['task_info'])
            env_name = env_name + '_' + config['task_info']['task']

        if 'linear' in args.__dict__.keys():
            if args.linear:
                from envs.wrappers import Linearize
                env = Linearize(env,env_name)
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]

    env.seed(args.seed)
    return env, env_name, action_dim, state_dim, traj, viewer
