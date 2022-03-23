import os
import yaml
import importlib
import gym

from envs import wrappers
from envs import minitaur


class BulletEnv:
    # 加载指定的bullet环境
    env_pakg = "envs.minitaur"

    def __init__(self, args):
        self.args = args
        self.config = {}
        self.max_length = None
        self.env_name = args.env_name
        self.env_class_name = "".join(list(map(lambda x: x.capitalize(), args.env_name.split("_"))))

    def build_env(self, **kwargs):
        env_module = importlib.import_module(name=f'.{self.env_name}', package=self.env_pakg)
        env_class = getattr(env_module, self.env_class_name)
        self.config = self.load_params_from_yaml()
        self.max_length = self.config.pop('max_length', None)
        self.random_max_steps = self.config.pop('random_max_steps', None)
        self.config.update(kwargs)
        # print("load env params: ", config)
        return self._create_environment(env_class(**self.config))

    def load_params_from_yaml(self):
        """
        从yaml配置文件加载环境的参数
        :param env_name_name:
        :return:
        """
        if not os.path.exists(os.path.join(os.path.dirname(__file__), f'{self.env_name}.yaml')):
            raise Exception("env is not exist.")

        env_config = {}
        with open(os.path.join(os.path.dirname(__file__), f'{self.env_name}.yaml'), 'r') as file:
            env_config = yaml.safe_load(file)

        return env_config

    def _create_environment(self, env):
        """Constructor for an instance of the environment.
        Args:
          config: Object providing configurations via attributes.

        Returns:
          Wrapped OpenAI Gym environment.
        """
        if self.max_length is not None:
            env = wrappers.LimitDuration(env, self.max_length)
        env = wrappers.RangeNormalize(env)
        env = wrappers.ClipAction(env)
        env = wrappers.ConvertTo32Bit(env)
        return env

