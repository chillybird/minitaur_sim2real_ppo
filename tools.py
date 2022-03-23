import os
import yaml


class Params:

    def __init__(self, file_name=None, from_dict=False, params=None):
        # 从yaml文件中加载参数，并保存为对象的属性
        # 实例变量
        if not file_name and not params:
            raise Exception("'file_name' or 'params' dict is needed to load params.")
        if not from_dict:
            self.load_params_from_yaml(file_name)
        else:
            self.load_params_from_dict(params)

    def load_params_from_yaml(self, file_name):
        with open(os.path.join(os.path.dirname(__file__), f'config/{file_name}.yaml'), 'r') as file:
            config = yaml.safe_load(file)
            self.__dict__.update(**config)

    def load_params_from_dict(self, params):
        self.__dict__.update(**params)

    def dump_params_to_yaml(self, file_name=None, params=None):
        if not file_name:
            print("file name can't be None.")
            return
        with open(os.path.join(os.path.dirname(__file__), f'config/{file_name}.yaml'), 'w') as file:
            if params is not None and isinstance(params, dict):
                yaml.dump(params, file)
            else:
                yaml.dump(self.__dict__, file)
