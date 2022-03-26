import time
import torch

from agents.ppo.model import ActorCritic
from envs import BulletEnv

env_name = "minitaur_reactive_env"
device = torch.device('cpu')
# 是否是连续的动作空间
has_continuous_action_space = True
random_seed = 0  #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0  #### set this to load a particular checkpoint num

directory = "saves" + '/' + env_name + '/'
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)

env_build_args = {
    'render': True,
    'use_signal_in_observation': True,
    'use_angle_in_observation': True,
}
env = BulletEnv(env_name).build_env(**env_build_args)
# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

action_std = 0.1

policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std, device=device)

# 从checkpoint加载网络的参数
policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

obs = env.reset()
total_reward = 0.0
total_steps = 0
frame_delay = 0
while True:
    env.render()
    obs_v = torch.FloatTensor(obs)
    action, _ = policy.act(obs_v)
    if has_continuous_action_space:
        action = action.numpy().flatten()
    else:
        action = action.item()
    obs, reward, done, _ = env.step(action)

    total_steps += 1
    total_reward += reward
    if done:
        obs = env.reset()
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))
        total_reward = 0
        total_steps = 0
    time.sleep(frame_delay)


