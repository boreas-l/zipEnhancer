# ------ coding : utf-8 ------
# @FileName     : inference_pipeline.py
# @Author       : lxc
# @Time         : 2025/3/4 14:48

from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
import torch

from models.zipenhancer import ZipenhancerDecorator
from utils import Config, audio_norm



class ANSZipEnhancer:
    """ANS (Acoustic Noise Suppression) Inference Pipeline"""
    SAMPLE_RATE = 8000

    def __init__(self, weight_path, **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        model_configs = kwargs.get('model')
        self.zipenhancer_obj = ZipenhancerDecorator(weight_path, **model_configs)
        self.model = self.zipenhancer_obj.model
        self.model.eval()
        print(f">>>ZipEnhancer语音降噪模型初始化完成！")
        self.stream_mode = kwargs.get('stream_mode', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, wav_data_path, **preprocess_params) -> Dict[str, Any]:
        audio_arr, sr = sf.read(wav_data_path)
        if sr != self.SAMPLE_RATE:
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        audio_arr = audio_norm(audio_arr)
        input_data = audio_arr.astype(np.float32)
        inputs = np.reshape(input_data, [1, input_data.shape[0]])
        return {'ndarray': inputs, 'nsamples': input_data.shape[0]}

    def forward(self, wav_data_path) -> Dict[str, Any]:
        inputs = self.preprocess(wav_data_path)
        ndarray = inputs['ndarray']
        if isinstance(ndarray, torch.Tensor):
            ndarray = ndarray.cpu().numpy()
        nsamples = inputs['nsamples']
        decode_do_segement = False
        # window = 16000 * 2  # 2s
        window = 8000 * 4
        stride = int(window * 0.75)
        print('inputs:{}'.format(ndarray.shape))
        b, t = ndarray.shape  # size()
        if t > window * 5:  # 10s
            decode_do_segement = True
            print('decode_do_segement')

        if t < window:
            ndarray = np.concatenate(
                [ndarray, np.zeros((ndarray.shape[0], window - t))], 1)
        elif decode_do_segement:
            if t < window + stride:
                padding = window + stride - t
                print('padding: {}'.format(padding))
                ndarray = np.concatenate(
                    [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
            else:
                if (t - window) % stride != 0:
                    # padding = t - (t - window) // stride * stride
                    padding = (
                        (t - window) // stride + 1) * stride + window - t
                    print('padding: {}'.format(padding))
                    ndarray = np.concatenate(
                        [ndarray,
                         np.zeros((ndarray.shape[0], padding))], 1)

        print('inputs after padding:{}'.format(ndarray.shape))
        with torch.no_grad():
            ndarray = torch.from_numpy(np.float32(ndarray)).to(self.device)
            b, t = ndarray.shape
            if decode_do_segement:
                outputs = np.zeros(t)
                give_up_length = (window - stride) // 2
                current_idx = 0
                while current_idx + window <= t:
                    # print('current_idx: {}'.format(current_idx))
                    print(
                        '\rcurrent_idx: {} {:.2f}%'.format(
                            current_idx, current_idx * 100 / t),
                        end='')
                    tmp_input = dict(noisy=ndarray[:, current_idx:current_idx
                                                   + window])
                    tmp_output = self.model(
                        tmp_input, )['wav_l2'][0].cpu().numpy()
                    end_index = current_idx + window - give_up_length
                    if current_idx == 0:
                        outputs[current_idx:
                                end_index] = tmp_output[:-give_up_length]
                    else:
                        outputs[current_idx
                                + give_up_length:end_index] = tmp_output[
                                    give_up_length:-give_up_length]
                    current_idx += stride
                print('\rcurrent_idx: {} {:.2f}%'.format(current_idx, 100))
            else:
                outputs = self.zipenhancer_obj.forward(dict(noisy=ndarray))['wav_l2'][0].cpu().numpy()
        outputs = (outputs[:nsamples] * 32768).astype(np.int16).tobytes()
        return {"output_pcm": outputs}

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        if 'output_path' in kwargs.keys():
            sf.write(
                kwargs['output_path'],
                np.frombuffer(inputs["output_pcm"], dtype=np.int16),
                self.SAMPLE_RATE)
        return inputs


if __name__ == '__main__':
    import time
    # 加载配置文件
    config_file_path = "./configs/configuration.json"
    cfg_dict, cfg_text = Config.file2dict(config_file_path)
    # 模型权重文件
    model_weight_path = "pytorch_model.bin"
    # 初始化降噪流程
    ans_zipenhancer = ANSZipEnhancer(model_weight_path, **cfg_dict)
    # wav文件路径
    start = time.perf_counter()
    wav_file_path = "./test_datas/speech_with_noise.wav"
    inference_res = ans_zipenhancer.forward(wav_file_path)
    use_time = time.perf_counter() - start
    print(f">>>inference res: {type(inference_res)}")
    print(f">>>推理耗时为：{use_time}s")
    # 保存降噪结果到本地，用于验证降噪效果
    # output_pcm_array = np.frombuffer(inference_res["output_pcm"], dtype=np.int16)
    # sf.write("output.wav", output_pcm_array, 8000)
    print("done")
