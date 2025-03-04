# ------ coding : utf-8 ------
# @FileName     : utils.py
# @Author       : lxc
# @Time         : 2025/3/4 15:16

"""工具函数合集"""
import os.path as osp
import yaml
import json



class Config:
    """用于加载配置文件"""
    @staticmethod
    def file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))  # 配置文件的绝对路径
        if not osp.exists(filename):
            raise ValueError(f'File does not exists {filename}')
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        # 打开配置文件
        if filename.endswith(('.yml', '.yaml')):
            with open(filename, 'r', encoding='utf-8') as infile:
                cfg_dict = yaml.safe_load(infile)
        else:
            with open(filename, 'r', encoding='utf-8') as infile:
                cfg_dict = json.load(infile)

        cfg_text = filename + '\n'
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        return cfg_dict, cfg_text


"""对音频信号进行归一化处理"""
def audio_norm(x):
    rms = (x**2).mean()**0.5
    scalar = 10**(-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean()**0.5
    scalarx = 10**(-25 / 20) / rmsx
    x = x * scalarx
    return x


if __name__ == '__main__':
    # 测试配置文件读取
    cfg_dict, cfg_text = Config.file2dict('./configs/configuration.json')
    print(cfg_dict)
    print(type(cfg_dict))
    print(cfg_text)
    print("done")
