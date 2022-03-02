import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio

from models import *
from utils.datasets import *
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
    parser.add_argument('--yaml_file', type=str, default='tomofun.yaml',
                    help='Specify yaml file (default: tomofun.yaml)')
    parser.add_argument('--model', type=str, default='Convnext_tiny',
                    help='Specify model (default: Convnext_tiny)')
    parser.add_argument('--model_saved_path', type=str, default='workdirs',
                    help='Specify the file path where the model is saved (default: workdirs)')
    parser.add_argument('--model_weights', type=str, default='best.pth',
                    help='Specify the weights where the model is saved (default: best.pth)')            
    parser.add_argument('--test_data', type=str, default='test_data',
                    help='Specify test data path (default: test_data)')

    parser.add_argument('--cuda_num', type=int, default=0,
                        help='number of CUDA (default: 0)')
    parser.add_argument('--local_rank', type=int, default=0,
                    help='number of local_rank for distributed training (default: 0)')
    parser.add_argument('--world-size', type=int, default=1,
                    help='number of nodes for distributed training (default: 1)')

    args = parser.parse_args()
    device, rank, world_size = init_distributed_mode(args.local_rank, args.cuda_num)
    print("device: {}, rank: {}, world_size: {}".format(device, rank, world_size))

    data_file, classes_info, data_set = load_data_info(args.yaml_file)
    classes_len, classes_names = classes_info[0], classes_info[1]
    
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
        mel_data = torch.Tensor(mel_data)

        total_mel_data.append(mel_data)

    model_saved_path = args.model_saved_path
    model_stats_path = os.path.join(model_saved_path, args.model_weights)

    if args.model == "ResNet18":
        model = ResNet18(1, classes_len)
    
    elif args.model == "ResNet34":
        model = ResNet34(1, classes_len)

    elif args.model == "ResNet50":
        model = ResNet50(1, classes_len)

    elif args.model == "ResNet101":
        model = ResNet101(1, classes_len)

    elif args.model == "ResNet152":
        model = ResNet152(1, classes_len)
        
    elif args.model == "SENet":
        model = SENet(classes_len)

    elif args.model == "DenseNet":
        model = DenseNet(classes_len)

    elif args.model == "Convnext_tiny":
        model = Convnext_tiny(classes_len)

    elif args.model == "Convnext_small":
        model = Convnext_tiny(classes_len)
        
    elif args.model == "Convnext_base":
        model = Convnext_base(classes_len)
        
    elif args.model == "Convnext_large":
        model = Convnext_tiny(classes_len)

    model.load_state_dict(torch.load(model_stats_path, map_location=torch.device(device)))
    model.eval()

    for i in range(len(total_mel_data)):
        data = total_mel_data[i].unsqueeze(0)

        outputs = model(data)
        preds = torch.max(outputs, 1)[1]

        print("Wav name: {}, Predict:{}".format(total_data_name[i], preds.item()))
              
    print("Predict finished.")