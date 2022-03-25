from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable 
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
import time
import timeit
from sklearn.metrics import precision_score , recall_score , confusion_matrix, f1_score, classification_report

from utils import models, miscellaneous, augmentations, datasets
from utils.datasets import VideoDataset
from torchvision import transforms

#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts



def parse_option_3d():
    parser = argparse.ArgumentParser()

    # Model related

    parser.add_argument('--imbalance', default=0, type=int, help='imbalance mode, 0 : Off, 1: On')

    parser.add_argument('--model_path', type=str, default='model_save/model_1.pth' , help='Model save path')

    #parser.add_argument('--', type=, default  = , help = '')
    # Data related
    parser.add_argument('--dataset', type=str, default='UCFCrimes', choices=['UCFCrimes','ucf101','hmdb51'], help = 'Dataset')
    parser.add_argument('--videofolder_path', type=str, default='dataset/Videos')
    parser.add_argument('--model_save_dir', type=str, default='various/modelsave')
    parser.add_argument('--sample_dir', type=str, default='various/sample')
    parser.add_argument('--result_dir', type=str, default='various/result')
    parser.add_argument('--num_class', type=int, default=2 , help = 'Number of class which we want to classify.')
    parser.add_argument('--preprocess_v2f', type=int, default=1, help='Check video preprocess, True : 1, False: 0')
    parser.add_argument('--preprocess_s', type=int, default=1 , help = 'Split train/test version')
    parser.add_argument('--n_frames', type=int, default=8 , help='number of input shape to 3D CNN')
    #parser.add_argument('--n_sub', type=int, default=, help='Number of segmentation(sub-shot) in videos')

    parser.add_argument('--val_split', type=int, default=1, help='Split val from train')
    # Train related


    parser.add_argument('--use_swa', default=False , help = 'SWA Chooser')

    parser.add_argument('--test_m', type=int, default=0 , help ='test_mode : 1, train_mode : 0')
    parser.add_argument('--enm', type=int, default=0, help='Ensemble mode On: 1, Ensemble mode Off : 0')
    parser.add_argument('--npy_path', type=str, default='npy_save/model_ucf_res_v1.npy' , help = 'npy save folder and file name')

    #parser.add_argument('--', type=, default  = , help = '')
    # Etc
    parser.add_argument('--csv_path', type=str, default='./csv_save/model_1.csv', help='csv save path')
    
    args = parser.parse_args()
    
    return args

def parse_option():
    parser = argparse.ArgumentParser()

    # All
    
    parser.add_argument('--set', type=str, default=None, help='2d or 3d')
    parser.add_argument('--img', type=int, default=224)


    # 3d
    parser.add_argument('--pretrain_path', type=str, default=None)



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

def train_model_3d(args, model):
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("\n Device is ", device)

    nEpochs = 70#args.epochs # Number of epochs for training , we will apply early stopping mode
    lr = args.lr # Learning rate, we will apply adaptive learning rate scheduler
    
    dataset = args.dataset
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    num_classes = args.nc # It will be binary or 14
    
    target_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise']
    
    #enm = args.enm

    #print(optimizer)
    print(model)
    print("Before start the training, delays for 5 seconds")
    time.sleep(2)
    print(' Total model parameters: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    """ 

    Not include summary writer

    """  

    train_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='train', clip_len=16,tm=0), batch_size=args.bs, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='val', clip_len=16, tm=1), batch_size=args.bs, shuffle=False, num_workers=8)
    

    trainval_loaders = {'train':train_dataloader, 'val':val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    print(trainval_sizes)

    early_stopping = EarlyStopping(patience=70, verbose=True) # Must check
    global_info = []
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_fs = 0.0
    best_model_wts = copy.deepcopy(model.state_dict()) # model weight parameter copy
    val_loss = 0.0
    for epoch in range(nEpochs):

        if epoch > 0:
            train_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='train', clip_len=16,tm=0), batch_size=args.bs, shuffle=True, num_workers=8)
            val_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='val', clip_len=16, tm=0), batch_size=args.bs, shuffle=True, num_workers=8)
    

            trainval_loaders = {'train':train_dataloader, 'val':val_dataloader}
            trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
            print(trainval_sizes)

        local_info = []

        for phase in ['train', 'val']:
            predlist=torch.zeros(0,dtype=torch.long, device='cpu')
            lbllist=torch.zeros(0,dtype=torch.long, device='cpu')   
            start_time = timeit.default_timer()
            
            # Reset the initial loss and corrects

            running_loss = 0.0
            running_corrects = 0.0

            # Split the model to train / eval mode
            if phase == 'train':
                model.train()

                scheduler.step(val_loss)
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                #print(inputs.shape)
                # Set All the parameter's gradient to zero
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)

                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                
          
                probs = nn.Softmax(dim=1)(outputs)


                preds = torch.max(probs, 1)[1]

                #print(preds) # preds
                #print(labels_class) # target
                
                
                #print(outputs.shape)
                
                loss = criterion(outputs, labels)
                #print("ho")
                if phase == 'train':
                    loss.backward() # back propataion is done when train phase
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]
            precision = precision_score(lbllist, predlist,average= "macro")
            recall = recall_score(lbllist, predlist,average= "macro") 
            fscore = 2*precision*recall/(precision + recall)#f1_score(lbllist, predlist, average="macro")
            #print("[{}] Epoch : {} / {} Loss : {} Acc : {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            if phase =='train':
                local_info.append(epoch_loss)
                ea = epoch_acc.cpu().numpy()
                local_info.append(ea)
                local_info.append(fscore)
                print("[{}] Epoch : {} / {} Loss : {} Acc : {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
                print("Precision : %0.3f"%precision,"Recall : %0.3f"%recall,"F-score: %0.3f"% fscore)
            elif phase =='val':
                local_info.append(epoch_loss)
                ea = epoch_acc.cpu().numpy()
                local_info.append(ea)
                local_info.append(fscore)
            #print("[{}] Epoch : {} / {} Loss : {} Acc : {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
        # Early stopping module
            if phase == 'val' and fscore > best_fs:
                best_fs = fscore
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best acc model
                model.load_state_dict(best_model_wts)
                print("save best acc model")
                torch.save(model, './models/best_fs_%s.pth'%(args.project))
        if phase == 'val':
            val_loss = running_loss / trainval_sizes[phase]
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping \n\n After Test mode begin")

                
                 
                break
    

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), args.model_path)
                print("Copy best model weights")
            print("[{}] Epoch : {} / {} Loss : {} Acc : {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            print("Precision : %0.3f"%precision,"Recall : %0.3f"%recall,"F-score: %0.3f"% fscore)
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        


    #print("Start test mode with best model wts")

    """
    model.eval()
    test_mode(args, enm, model)
    """


    data = pd.DataFrame(global_info, columns= ['train_loss', 'train_acc','train_fscore', 'val_loss', 'val_acc','val_fscore'])
    data.to_csv('./csv_save/%s.csv'%(args.project), header=True, index=True)
    stop_time = timeit.default_timer()
    print(str(stop_time-start_time))


    return model
    
    
    
def test_mode(args, enm, model):
    tm = 1
    dataset = args.dataset
    num_classes = args.num_class
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss() # Standard crossentropy loss for classification
    criterion.to(device)

    count = 0
    fusion = []
    if enm ==0:
        test_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='test', clip_len=16,tm=1), batch_size=16, shuffle=True, num_workers=8)
        test_size = len(test_dataloader.dataset)
        print(test_size)
        start_time = timeit.default_timer()            
 
        running_loss_t = 0.0
        running_corrects_t = 0.0
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            with torch.no_grad():
                outputs = model(inputs)
        
            if num_classes ==13 or num_classes ==51 or num_classes == 101:
                probs = nn.Softmax(dim=1)(outputs)
            elif num_classes ==2:
                probs = nn.Sigmoid()(outputs) 

            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            if preds == labels:
                count = count + 1
            outputs_temp = np.append(outputs, int(labels))
            fusion.append(output)

            running_loss_t += loss.item() * inputs.size(0)
            running_corrects_t += torch.sum(preds == labels.data)
    
        epoch_loss_t = running_loss_t / test_size
        epoch_acc_t = running_corrects_t.double() / test_size
        if num_classes ==101:
            fusion = fusion[:3783]
        elif num_classes ==51:
            fusion = fusion[:1530]
        np.save(args.npy_path, fusion)
        print("[Test] Loss : {} Acc: {}".format(epoch_loss_t, epoch_acc_t))
        stop_time = timeit.default_timer()
        print("Execution time : " + str(stop_time - start_time ) + "\n")
    elif enm == 1: # 넣을지 말지 모름 - 미완성 코드
        start_time = timeit.default_timer()

        loss_sum = 0.0
        acc_sum = 0.0
        for it in range(5):

            test_dataloader = DataLoader(VideoDataset(args,dataset=dataset, split='val', clip_len=16), batch_size=16, shuffle=True, num_workers=8, tm=0)
            test_size = len(test_dataloader.dataset)
            print(test_size)
            
 
            running_loss_t = 0.0
            running_corrects_t = 0.0


            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                with torch.no_grad():
                    outputs = model(inputs)
        
                if num_classes ==13:
                    probs = nn.Softmax(dim=1)(outputs)
                elif num_classes ==2:
                    probs = nn.Sigmoid()(outputs) 
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss_t += loss.item() * inputs.size(0)
                running_corrects_t += torch.sum(preds == labels.data)
    
        epoch_loss_t = running_loss_t / test_size
        epoch_acc_t = running_corrects_t.double() / test_size
    
        print("[Test] Loss : {} Acc: {}".format(epoch_loss_t, epoch_acc_t))
        stop_time = timeit.default_timer()
        print("Execution time : " + str(stop_time - start_time ) + "\n")
        
        
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
                
                #print(labels, labels.shape)
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

                    inputs_temp, labels_list_a, labels_list_b,random_jit,random_jit2 = augmentations.halfmix_out_jit_lineout(args, inputs, labels, rand_index, phase, labels_new_list)             
                    
                    
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                
                    if phase == 'train':
                
                        outputs = model(inputs_temp)
                        #print(outputs, outputs.shape)
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
    

    if args.set == '3d':
        

        """
        model = models.resnext101(
                num_classes=args.nc,
                shortcut_type='B',
                cardinality=32,
                sample_size=args.img,
                sample_duration=16)
        pretrain = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
        
        model.load_state_dict(pretrain['state_dict'], strict=False)
        """
        model = models.model_set(args)
        
        #model.module_fc = nn.Linear(model.module.fc.in_features, args.nc)
        device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
        #model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        model.to(device)


        optimizer = miscellaneous.optimizer_set(args, model)

        # Load Loss function

        criterion = miscellaneous.loss_set(args)#

        # Load Learning Rate Scheduler

        scheduler = miscellaneous.scheduler_set(args, optimizer) 
        
        model = train_model_3d(args, model)
    else:

   

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
    
    
    
    
    
    
    
    
    
    
    
    
