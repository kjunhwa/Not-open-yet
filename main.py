from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable 
import torch.nn.functional as F

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from pytorchtools import EarlyStopping
import argparse
from tqdm import tqdm
import natsort as nt
import shutil
import timm 
import random

from sklearn.metrics import precision_score , recall_score , confusion_matrix, f1_score, classification_report

from utils import models, miscellaneous, augmentations, datasets



#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='raf', help = 'dataset')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--models', type=str)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--w', type=int, default=0, help='0 : no weight, 1 : weight')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--aug', type=int, default=0, help = '0 : no aug, 1 : aug')
    parser.add_argument('--e', type=int, default=15, help = 'early stopping')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--mix_prob', default = 0.0,type=float, help='mix prob')
    parser.add_argument('--project',default='project', type=str)
    parser.add_argument('--nc', default = 7, type=int)
    parser.add_argument('--bs', default = 64, type=int)
    parser.add_argument('--nw', default = 0, type=int)
    parser.add_argument('--mode', default=0, type=int, help='select mode 0 : None, mode 1 : half mix, mode 2 : half out mode 3 : half mix jit, mode 4 : lineout, mode 5 : mixoutjitline')
    parser.add_argument('--imbalance', default=0, type=int, help='imbalance mode, 0 : Off, 1: On')
    args = parser.parse_args()
    return args


def train_model(model, args,dataset_sizes ,num_epochs=20):
    
    target_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise']
    
    global_info = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    best_acc =0.0
    best_fs = 0.0
    early_stopping = EarlyStopping(patience=args.e, verbose=True)
    for epoch in range(num_epochs):


        local_info = []
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #print("Current Learning rate", scheduler.get_last_lr())
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        
            y_pred = []
            y_true = []
            predlist=torch.zeros(0,dtype=torch.long, device='cpu')
            lbllist=torch.zeros(0,dtype=torch.long, device='cpu')   
            if phase == 'train':
                
                model.train()  # Set model to training mode
            else:
                
                model.eval()   # Set model to evaluate mode
                if epoch >0:
                #    print(val_loss)
                    scheduler.step()
                    
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device) # 16 

                labels = labels.to(device)
                

                # imbalance mode set
                
                if args.imbalance == 1:
                
                    labels_new_list = augmentations.imbalance_set(args, labels)
                else:
                    labels_new_list = []
                if args.imbalance == 1:
                
                    rand_index = torch.randperm(len(labels_new_list)).cuda()
                else:
                    rand_index = torch.randperm(inputs.size()[0]).cuda() # batch 안에서 랜덤하게 뽑음
                

                if args.mode == 0:
                    labels = F.one_hot(labels.long(), args.nc).type(torch.float32)
                    inputs_temp = inputs
                    
                elif args.mode == 1 : # half mix
                    
                    inputs_temp, labels_list_a, labels_list_b = augmentations.halfmix(args, inputs, labels, rand_index, phase, labels_new_list) 
                
                elif args.mode == 2: # half out
                
                    labels = F.one_hot(labels.long(), args.nc).type(torch.float32) # val에서 에러
                    
                    inputs_temp = augmentations.halfout(args, inputs, labels, rand_index, phase, labels_new_list) 
                elif args.mode == 3: # half mix Jittering

                    
                    inputs_temp, labels_list_a, labels_list_b,random_jit,random_jit2 = augmentations.halfmix_jit(args, inputs, labels, rand_index, phase, labels_new_list)
                elif args.mode == 4: # lineout
                    labels = F.one_hot(labels.long(), args.nc).type(torch.float32)
                    inputs_temp = augmentations.lineout(args, inputs, labels, rand_index, phase, labels_new_list) 
                    
                elif args.mode == 5:

                    inputs_temp, labels_list_a, labels_list_b,random_jit,random_jit2 = augmentations.halfmix_out_jit(args, inputs, labels, rand_index, phase, labels_new_list)             
                    
                    
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                
                    if phase == 'train':
                
                        outputs = model(inputs_temp)
                    else:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(inputs_temp)
                            

                    _, preds = torch.max(outputs, 1)
                    
                    if args.mode == 0:
                    
                        loss = criterion(outputs, labels)
                        
                    elif args.mode == 1:
                        labels = F.one_hot(labels.long(), args.nc).type(torch.float32)
                        labels_list_a =  F.one_hot(labels_list_a.long(), args.nc).type(torch.float32)
                        labels_list_b =  F.one_hot(labels_list_b.long(), args.nc).type(torch.float32)
                        loss_a = criterion(outputs, labels_list_a)*0.5
                        loss_b = criterion(outputs, labels_list_b)*0.5
                        loss =  loss_a+ loss_b
                        
                    elif args.mode == 2 or args.mode == 4:
                    
                        loss = criterion(outputs, labels) 
                    elif args.mode == 3 or args.mode == 5:
                        labels = F.one_hot(labels.long(), args.nc).type(torch.float32)
                        labels_list_a =  F.one_hot(labels_list_a.long(), args.nc).type(torch.float32)
                        labels_list_b =  F.one_hot(labels_list_b.long(), args.nc).type(torch.float32)
                        loss_a = criterion(outputs, labels_list_a)*random_jit
                        loss_b = criterion(outputs, labels_list_b)*random_jit2
                        loss =  loss_a+ loss_b
                    #loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, labels_class = torch.max(labels, 1)
                #print(preds) # preds
                #print(labels_class) # target
                running_corrects += torch.sum(preds == labels_class.data)
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,labels_class.view(-1).cpu()])
                #y_pred.append(preds.cpu().numpy().astype(int))
                #y_true.append(labels_class.data.cpu().numpy().astype(int))
                # P R -> F SCORE
                
                

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'val':
                val_loss = running_loss / dataset_sizes['val']
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # P R F SCORE TOTAL
            #y_true = y_true
            #y_pred = y_pred
            precision = precision_score(lbllist, predlist,average= "macro")
            recall = recall_score(lbllist, predlist,average= "macro") 
            fscore = 2*precision*recall/(precision + recall)#f1_score(lbllist, predlist, average="macro")
            
            if phase == 'train':
                local_info.append(epoch_loss)
                ea = epoch_acc.cpu().numpy()
                local_info.append(ea)
                local_info.append(fscore)
            else:
                local_info.append(epoch_loss)
                ea = epoch_acc.cpu().numpy()
                local_info.append(ea)
                local_info.append(fscore)

            #print(classification_report(lbllist, predlist, target_names=target_name) )               
            print("Precision : %0.3f"%precision,"Recall : %0.3f"%recall,"F-score: %0.3f"% fscore)
            #print('precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}'.format(precision, recall, fscore))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best acc model
                model.load_state_dict(best_model_wts)
                print("save best acc model")
                torch.save(model, './models/best_acc_%s.pth'%(args.project))
            if phase == 'val' and fscore > best_fs:
                best_fs = fscore
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best acc model
                model.load_state_dict(best_model_wts)
                print("save best fscore model")
                torch.save(model, './models/best_fscore_%s.pth'%(args.project))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts_loss = copy.deepcopy(model.state_dict())
                # save best loss model
                model.load_state_dict(best_model_wts_loss)
                print("save best loss model")
                torch.save(model, './models/best_loss_%s.pth'%(args.project))
                
        lr_get = get_lr(optimizer)
        
        print("Current learning rate : {:.8f}".format(lr_get))
        
        global_info.append(local_info)
        
        if phase =='val':
            early_stopping(epoch_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        #scheduler.step(val_loss)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    #model.load_state_dict(best_model_wts)
    data = pd.DataFrame(global_info, columns = ['train_loss', 'train_acc','train_fscore', 'val_loss', 'val_acc','val_fscore'])
    data.to_csv('./csv_save/%s.csv'%(args.project), header=True, index=True)
    
    # save last model
    print("save last model")
    torch.save(model, './models/last_%s.pth'%(args.project))  
    return model.load_state_dict(best_model_wts), best_acc, model.load_state_dict(best_model_wts_loss), model
    
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




if __name__ == '__main__':


    args = parse_option()
   

    save_name = args.models
    name = args.dataset
    pat = args.patience

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    
    data_dir = os.path.join(os.getcwd(), 'dataset/%s'%name)

    data_transforms = augmentations.augmentation_set(args)
    dataloaders, dataset_sizes, class_names = datasets.LoadDataset(args, data_dir, data_transforms)

    print(device)
    
    # Load Pretrained Models

    model_ft = models.model_set(args)
    model_ft = model_ft.to(device)

    # Load optimizer

    optimizer = miscellaneous.optimizer_set(args, model_ft)

    # Load Loss function

    criterion = miscellaneous.loss_set(args)#

    # Load Learning Rate Scheduler

    scheduler = miscellaneous.scheduler_set(args, optimizer) 


    model_ft, best_acc, model_ft_loss, model_ft_last = train_model(model_ft, args,dataset_sizes, num_epochs=70)
    
    
    
    
    
    
    
    
    
    
    
    
