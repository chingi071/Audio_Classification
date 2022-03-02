import os
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchaudio
from utils.utils import *

def data_padding(data, input_length):
    if data.shape[-1] > input_length:
        max_offset = data.shape[-1] - input_length
        offset = np.random.randint(max_offset)
        data = data[:, :, offset:(input_length+offset)]

    else:
        max_offset = input_length - data.shape[-1]
        offset = max_offset//2
        data = np.pad(data, ((0, 0), (0, 0), (offset, max_offset - offset)), "constant")

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()          
    parser.add_argument('--test_data', type=str, default='test_data',
                    help='Specify test data path (default: test_data)')
    parser.add_argument('--cuda_num', type=int, default=0,
                     help='number of CUDA (default: 0)')
    parser.add_argument('--local_rank', type=int, default=0,
                 help='number of local_rank for distributed training (default: 0)')

    args = parser.parse_args()
    device, rank, world_size = init_distributed_mode(args.local_rank, args.cuda_num)
    print("device: {}, rank: {}, world_size: {}".format(device, rank, world_size))

    to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                        sample_rate=16000, n_fft=500, n_mels=64,
                        hop_length=160, f_min=0, f_max=8000)
    input_length = 500

    total_mel_data = []
    total_data_name = []
    for i in os.listdir(args.test_data):
        total_data_name.append(i)
        wav_path = os.path.join(args.test_data, i)

        data, sr = torchaudio.load(wav_path)
        data = torchaudio.transforms.Resample(sr, 16000)(data)
        mel_spec = to_mel_spectrogram(data)
        #mel_spec = torchaudio.transforms.TimeMasking(time_mask_param=80)(mel_spec)
        log_mel_spec = (mel_spec + torch.finfo(torch.float).eps).log()
        mel_data = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + np.finfo(np.float64).eps)
        mel_data = data_padding(mel_data, input_length)

        total_mel_data.append(mel_data)

    session = ort.InferenceSession('./model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for i in range(len(total_mel_data)):
        data = total_mel_data[i].unsqueeze(0).numpy()
        result = session.run([output_name], {input_name: data})
        
        preds = int(np.argmax(np.array(result).squeeze(), axis=0))        
        print("Wav name: {}, Predict:{}".format(total_data_name[i], preds))

    print("Predict finished.")