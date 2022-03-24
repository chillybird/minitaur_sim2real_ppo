import os
from datetime import datetime
import numpy as np
import torch

from agents.ppo.common import PPO
from init import train_params, output_params_info, train_init
from envs import BulletEnv

max_length = 1000


def train():
    print("============================================================================================")

    env_name = 'minitaur_trotting_env'
    env_build_args = {
        'render': True,
        'use_signal_in_observation': True,
        'use_angle_in_observation': True,
    }
    env = BulletEnv(env_name).build_env(**env_build_args)

    output_params_info(env_name)
    writer, checkpoint_path = train_init(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    if train_params.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if train_params.random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", train_params.random_seed)
        torch.manual_seed(train_params.random_seed)
        env.seed(train_params.random_seed)
        np.random.seed(train_params.random_seed)

    print("============================================================================================")
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, train_params.lr_actor, train_params.lr_critic,
                    train_params.gamma, train_params.K_epochs, train_params.eps_clip,
                    train_params.has_continuous_action_space, train_params.action_std, use_gpu=True)

    if os.path.exists(checkpoint_path):
        print("Load pretrained model from checkpoint:", checkpoint_path)
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while train_params.loop_flag or time_step <= train_params.max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, train_params.max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % train_params.update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if train_params.has_continuous_action_space and time_step % train_params.action_std_decay_freq == 0:
                ppo_agent.decay_action_std(train_params.action_std_decay_rate, train_params.min_action_std)

            # log in logging file
            if time_step % train_params.log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                if writer:
                    writer.add_scalar("averge_rewad", log_avg_reward, time_step)
                    # writer.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % train_params.print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % train_params.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
    if writer:
        writer.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
