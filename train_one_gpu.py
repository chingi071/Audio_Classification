import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
import torchvision.models as models
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from tensorboardX import SummaryWriter
from models import *
from utils.datasets import *
from utils.utils import *

def train_epoch(epoch, model, lr_scheduler, optimizer, criterion):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        target_value = torch.max(target, 1)[1]

        optimizer.zero_grad()
        
        outputs = model(data)
        preds = torch.max(outputs, 1)[1]
        
        loss = criterion(outputs, target_value)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += torch.eq(preds, target_value).sum().item()
                
        '''
        if batch_idx % 1 == 0:
            print("Train Epoch: {}/{} [iter:{}/{}], acc:{}, loss:{}\
                    ".format(epoch, num_epochs-1, batch_idx+1, len(train_loader),
                       running_acc / float((batch_idx + 1) * batch_size),
                       running_loss / float((batch_idx + 1) * batch_size)))
        '''

    lr_scheduler.step()
    train_acc_value = running_acc/len(dataset_train)
    train_loss_value = running_loss/len(dataset_train)

    return train_acc_value, train_loss_value
    
def valid_epoch(epoch, model, criterion, model_saved_path, classes_names):
    global best_valid_loss
    model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    batch_target = []
    batch_pred = []
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            target_value = torch.max(target, 1)[1]
            batch_target.append(target_value)
            
            outputs = model(data)
            preds = torch.max(outputs, 1)[1]
            batch_pred.append(preds)

            loss = criterion(outputs, target_value)
            
            running_loss += loss.item() 
            running_acc += torch.sum(preds == target_value)
            
    valid_acc_value = running_acc/len(dataset_valid)
    valid_loss_value = running_loss/len(dataset_valid)   
    batch_target = torch.cat(batch_target)
    batch_pred = torch.cat(batch_pred)
        
    if epoch % 10 == 0:
        model_path = os.path.join(model_saved_path, "epoch_%d.pth" % epoch)
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, model_path)
        
    if valid_loss_value <= best_valid_loss:
        model_path = os.path.join(model_saved_path, "best.pth")
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, model_path)
        best_valid_loss = valid_loss_value
        confusion_matrix_plot(batch_target, batch_pred, classes_names, model_saved_path)
            
    return valid_acc_value, valid_loss_value
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='tomofun.yaml',
                        help='Specify yaml file (default: tomofun.yaml)')
    parser.add_argument('--model', type=str, default='ResNet18',
                        help='Specify model (default: ResNet18)')
    parser.add_argument('--model_saved_path', type=str, default='workdirs',
                        help='Specify the file path where the model is saved (default: workdirs)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Specify optimizer for training (default: AdamW)')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='Specify lr_scheduler for training (default: CosineAnnealingLR)')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                        help='Specify loss function for training (default: CrossEntropyLoss)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
                        
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Specify epoch for training (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify batch size for training (default: 256)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Specify prefetch factor for training (default: 2)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Specify num workers for training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Specify training learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Specify weight decay for training (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Specify momentum for training (default: 0.9)')
    parser.add_argument('--cuda_num', type=int, default=0,
                        help='number of CUDA (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    
    args = parser.parse_args()
    device, rank, world_size = init_distributed_mode(0, args.cuda_num)
    print("device: {}, rank: {}, world_size: {}".format(device, rank, world_size))
    
    data_file, classes_info, data_set = load_data_info(args.yaml_file)
    train_file, valid_file = data_file[0], data_file[1]
    classes_len, classes_names = classes_info[0], classes_info[1]
    train_set, valid_set = data_set[0], data_set[1]
    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    model_saved_path = args.model_saved_path
    best_valid_loss = np.Inf
    
    try:
        if not os.path.exists(model_saved_path):
            os.mkdir(model_saved_path)
    except:
        pass
    
    random_seed(args.seed)
    writer = SummaryWriter()
    
    '''
    apply_augmentation = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
        TimeStretch(min_rate=1.25, max_rate=1.7, p=1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=1, max_fraction=1, rollover=True, p=0.5),
    ])
    '''
    
    dataset_train = audio_Dataset(train_set, train_file, classes_names)
    dataset_valid = audio_Dataset(valid_set, valid_file, classes_names)

    train_loader = DataLoader(dataset_train, batch_size, prefetch_factor=2,num_workers=4, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size, prefetch_factor=2,num_workers=4, shuffle=False)
        
    if args.model == "ResNet18":
        model = ResNet18(1, classes_len).to(device)
    
    elif args.model == "ResNet34":
        model = ResNet34(1, classes_len).to(device)

    elif args.model == "ResNet50":
        model = ResNet50(1, classes_len).to(device)

    elif args.model == "ResNet101":
        model = ResNet101(1, classes_len).to(device)

    elif args.model == "ResNet152":
        model = ResNet152(1, classes_len).to(device)
        
    elif args.model == "SENet":
        model = SENet(classes_len).to(device)
        
    elif args.model == "DenseNet":
        model = DenseNet(classes_len).to(device)
        
    elif args.model == "Convnext_tiny":
        model = Convnext_tiny(classes_len).to(device)
        
    elif args.model == "Convnext_small":
        model = Convnext_tiny(classes_len).to(device)
        
    elif args.model == "Convnext_base":
        model = Convnext_base(classes_len).to(device)
        
    elif args.model == "Convnext_large":
        model = Convnext_tiny(classes_len).to(device)

    # ===== optimizer =====
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay) 
    
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # ===== lr_scheduler =====
    if args.lr_scheduler == "StepLR":
        lr_scheduler_values = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
        
    elif args.lr_scheduler == "ExponentialLR":
        lr_scheduler_values = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler_values = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    elif args.lr_scheduler == "CyclicLR":
        lr_scheduler_values = lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    
    # ===== loss_function =====
    if args.loss_function == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
        
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        
    # =========================

    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    for epoch in range(num_epochs):
        train_acc_value, train_loss_value = train_epoch(epoch, model, lr_scheduler_values, optimizer, criterion)
        
        train_acc.append(train_acc_value)
        train_loss.append(train_loss_value)
        
        print("Train Epoch: {}/{} Training_Loss: {} Training_acc: {:.2f}\
               ".format(epoch, num_epochs-1, train_loss[epoch], train_acc[epoch]))
        
        valid_acc_value, valid_loss_value = valid_epoch(epoch, model, criterion, model_saved_path, classes_names)
                
        valid_acc.append(valid_acc_value)
        valid_loss.append(valid_loss_value)
        
        writer.add_scalars("Accuracy", {"Train":float(train_acc_value), "Valid":float(valid_acc_value)}, epoch)
        writer.add_scalars("loss", {"Train":float(train_loss_value), "Valid":float(valid_loss_value)}, epoch)
        
        print("Valid Epoch: {}/{} Valid_Loss: {} Valid_acc: {:.2f}\
               ".format(epoch, num_epochs-1, valid_loss[epoch], valid_acc[epoch]))
        print('--------------------------------')
        
        writer.close()
        
    visualization(num_epochs, train_acc, valid_acc, 'Accuracy', model_saved_path)
    visualization(num_epochs, train_loss, valid_loss, 'Loss', model_saved_path)

    print('Finished Training.')    

