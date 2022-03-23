"""
保存训练所需要的参数
"""
import os
from tools import Params


# 将训练的超参数转换为对象的属性进行调用
max_ep_len = 1000
train_params = Params(from_dict=True, params={

    # 'env_name': 'minitaur_reactive_env',
    'env_name': 'minitaur_trotting_env',
    'has_continuous_action_space': True,  # continuous action space; else discrete
    'open_writer': True,   # 是否使用 SummaryWriter
    'loop_flag': True, # 是否停止启用最大训练时间步数

    'max_ep_len':max_ep_len,                   # max timesteps in one episode
    'max_training_timesteps': int(3e6),   # break training loop if timeteps > max_training_timesteps

    'print_freq': max_ep_len * 10,        # print avg reward in the interval (in num timesteps)
    'log_freq': max_ep_len * 2,           # log avg reward in the interval (in num timesteps)
    'save_model_freq': int(1e5),          # save model frequency (in num timesteps)

    'action_std': 0.6,                    # starting std for action distribution (Multivariate Normal)
    # 动作分布的方差越小，就表示策略越确定
    'action_std_decay_rate': 0.05,        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    'min_action_std': 0.1,               # minimum action_std (stop decay after action_std <= min_action_std)
    'action_std_decay_freq': int(2.5e5),  # action_std decay frequency (in num timesteps)

    ################ PPO hyperparameters ################
    'update_timestep': max_ep_len * 4,      # update policy every n timesteps
    'K_epochs': 80,               # update policy for K epochs in one PPO update

    'eps_clip': 0.2,          # clip parameter for PPO
    'gamma': 0.99,            # discount factor

    'lr_actor': 0.0003,       # learning rate for actor network
    'lr_critic': 0.001,       # learning rate for critic network

    'random_seed': 0,        # set random seed if required (0 = no random seed)
})


def output_params_info():
    """
    print all hyperparameters
    :return:
    """
    print("--------------------------------------------------------------------------------------------")
    print("training environment name : " + train_params.env_name)
    print("max training timesteps : ",  'infinity' if train_params.loop_flag else train_params.max_training_timesteps)
    print("max timesteps per episode : ", train_params.max_ep_len)

    print("model saving frequency : " + str(train_params.save_model_freq) + " timesteps")
    print("log frequency : " + str(train_params.log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(train_params.print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    if train_params.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", train_params.action_std)
        print("decay rate of std of action distribution : ", train_params.action_std_decay_rate)
        print("minimum std of action distribution : ", train_params.min_action_std)
        print(
            "decay frequency of std of action distribution : " + str(train_params.action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(train_params.update_timestep) + " timesteps")
    print("PPO K epochs : ", train_params.K_epochs)
    print("PPO epsilon clip : ", train_params.eps_clip)
    print("discount factor (gamma) : ", train_params.gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", train_params.lr_actor)
    print("optimizer learning rate critic : ", train_params.lr_critic)


def train_init():
    """
    获取log文件对象和checkpoint路径
    :return:
    """
    writer = None
    ###################### logging ######################
    if train_params.open_writer:

        from tensorboardX import SummaryWriter

        log_dir = 'runs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create tensorboard SummaryWriter for each run
        writer = SummaryWriter(comment='PPO_' + train_params.env_name + "_" + str(run_num))

        print("current logging run number for " + train_params.env_name + " : ", run_num)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "saves"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + train_params.env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(train_params.env_name, train_params.random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    return writer, checkpoint_path


writer, checkpoint_path = train_init()
