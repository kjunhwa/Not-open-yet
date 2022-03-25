from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import natsort as nt
import timm 
import random
#from utils import models, miscellaneous, augmentations, datasets, my_f1_score, make_folder
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import warnings
import glob
warnings.filterwarnings('ignore')

def refine_frames_paths(frames, length):
    if (len(frames) > length) and np.abs(len(frames) - length) == 1:
        length = len(frames)
    frames_ids = [int(frame.replace("\\", "/").split('/')[-1].split('.')[0]) - 1 for frame in frames]
    if len(frames) == length:
        return frames
    else:
        extra_frame_ids = []
        prev_frame_id = frames_ids[0]
        for i in range(length):
            if i not in frames_ids:
                extra_frame_ids.append(prev_frame_id)
            else:
                prev_frame_id = i
        frames_ids.extend(extra_frame_ids)
        frames_ids = sorted(frames_ids)
        prefix = '/'.join(frames[0].split('/')[:-1])
        return_frames = [prefix + '/{0:05d}.jpg'.format(id + 1) for id in frames_ids]
        return return_frames
        
def name_trans(name):
    return '{:0>5}'.format(name) # 1 -> 00001
    
def melspec_f(samples, n_mels = 128):
    
    S = librosa.feature.melspectrogram(samples.astype(np.float16), n_mels)

    log_S = librosa.power_to_db(S, ref=np.max)
    #print("shape : ", log_S.shape)
    return log_S
    
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled
    
def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_folder', type=str, default='./save/', help = 'dataset')
    parser.add_argument('--img_path', type=str, default='E:/Daehakwon/experiment/abaw/crop/batch/', help = 'dataset')
    parser.add_argument('--aud_path', type=str, default='E:/Daehakwon/experiment/abaw/dataset3/audio/', help = 'dataset')
    parser.add_argument('--vid_path', type=str, default='E:/Daehakwon/experiment/abaw/dataset3/batch/', help = 'dataset')
    parser.add_argument('--test_txt', type=str, default='./list.txt', help = 'dataset')
    parser.add_argument('--weight', type=float, default=0.7)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    GT_list = [1, 2, 3, 4, 0, 7, 5, 6]
    # Test video list 
    f = open(args.test_txt, "r")
    list_names = f.readlines()
    f.close()
    device = "cuda:1"
    #Image Transform
    trans = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
    # Model load
    cnn_2d = torch.load("image_fs_best.pth", map_location="cuda:1")
    print("2D Model load complete!")
    cnn_3d = torch.load("video_fs_best.pth", map_location="cuda:1")
    print("3D Model load complete!")
    cnn_aud = torch.load("audio_fs_best.pth", map_location="cuda:1")
    print("Audio Model load complete!")
    softmax = nn.Softmax()
    # Prediction
    print("Inference Start")
    plt.figure(figsize=(8, 8))
    with torch.no_grad():              
        for video in tqdm(list_names):            
            list_names = video.strip() # Main Name (ex, 2-30-640x360)
                        
            # Video File Load for total frame number
            frames_paths = sorted(glob.glob(os.path.join(args.img_path, list_names, '*.jpg')))
            
            if '_left' in list_names:
                list_name = list_names[:-5]
            elif '_right' in list_names:
                list_name = list_names[:-6]
            else:
                list_name = list_names
                
            vid_path = glob.glob(os.path.join(args.vid_path, list_name + ".*"))[0]
            my_vid = cv2.VideoCapture(vid_path)  
                
            # total frame number
            vid_len = int(my_vid.get(7)) + 1             
            frames_paths = refine_frames_paths(frames_paths, vid_len)
            vid_len = len(frames_paths)            
            
            mok1, nam1 = divmod(vid_len, 60)
            
            if nam1 < 30:
                return_num = vid_len - 30
                flag = True
            else:
                return_num = vid_len - 60 # 나머지 값
                flag = False
                
            my_vid.release() # Remove Video to save memory
            
            # Audio File Load
            aud_path = args.aud_path + list_name + ".wav" 
            y, sr = librosa.load(aud_path, sr=16000)
            
             # Image Folder Name
            img_path = args.img_path + list_names
            
            save_pred0 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred1 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred2 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred3 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred4 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred5 = torch.zeros(60,8).to(device) # Store total prediction
            save_pred6 = torch.zeros(60,8).to(device) # Store total prediction
            
            save_img = [] # Store imgs for 3D-CNN            
            
            if flag:
                total_rest_pred0 = torch.zeros(30,8).to(device)
                total_rest_pred1 = torch.zeros(30,8).to(device)
                total_rest_pred2 = torch.zeros(30,8).to(device)
                total_rest_pred3 = torch.zeros(30,8).to(device)
                total_rest_pred4 = torch.zeros(30,8).to(device)
                total_rest_pred5 = torch.zeros(30,8).to(device)
                total_rest_pred6 = torch.zeros(30,8).to(device)
                rest_img1 = []
            else:
                total_rest_pred0 = torch.zeros(60,8).to(device)
                total_rest_pred1 = torch.zeros(60,8).to(device)
                total_rest_pred2 = torch.zeros(60,8).to(device)
                total_rest_pred3 = torch.zeros(60,8).to(device)
                total_rest_pred4 = torch.zeros(60,8).to(device)
                total_rest_pred5 = torch.zeros(60,8).to(device)
                total_rest_pred6 = torch.zeros(60,8).to(device)
                rest_img1 = []
                rest_img2 = []
                
            rest_cnt = 0
                            
            txt_name = list_names + ".txt" # Result text
            save_name0 = os.path.join(args.save_folder, "0", txt_name)     
            with open(save_name0, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성
                
            save_name1 = os.path.join(args.save_folder, "1", txt_name)     
            with open(save_name1, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성   
                
            save_name2 = os.path.join(args.save_folder, "2", txt_name)     
            with open(save_name2, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성

            save_name3 = os.path.join(args.save_folder, "3", txt_name)     
            with open(save_name3, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성
                
            save_name4 = os.path.join(args.save_folder, "4", txt_name)     
            with open(save_name4, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성                

            save_name5 = os.path.join(args.save_folder, "5", txt_name)     
            with open(save_name5, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성

            save_name6 = os.path.join(args.save_folder, "6", txt_name)     
            with open(save_name6, "w") as fd: # txt 생성
                fd.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n') # 첫 줄 작성
               
            for i in tqdm(range(vid_len)): # Video 전체 길이만큼 for문 반복
                temp_img = name_trans(str(i+1)) # Translate name (1 -> 00001)
                full_img_path = img_path + '/' + temp_img + '.jpg'
                #print(full_img_path)
                if os.path.isfile(full_img_path): # 해당 이미지가 존재하는지 확인
                    temp_img = Image.open(full_img_path) # Image load
                    temp_img = trans(temp_img) # Transform for input
                    temp_img = torch.unsqueeze(temp_img, 0)
                    temp_img = temp_img.to(device)                        
                    save_img.append(temp_img) # Save Image                        
                    pred_2d = torch.squeeze(softmax(cnn_2d(temp_img)), 0) # 2D CNN inference   
                    
                    save_pred0[i%60,:] += pred_2d # Save 2D CNN Result                               
                    save_pred1[i%60,:] += pred_2d # Save 2D CNN Result                               
                    save_pred4[i%60,:] += pred_2d # Save 2D CNN Result                               
                    save_pred5[i%60,:] += pred_2d # Save 2D CNN Result
                    
                    if i >= return_num:
                        if flag:
                            total_rest_pred0[rest_cnt%30,:] = pred_2d
                            total_rest_pred1[rest_cnt%30,:] = pred_2d
                            total_rest_pred4[rest_cnt%30,:] = pred_2d
                            total_rest_pred5[rest_cnt%30,:] = pred_2d
                            rest_img1.append(temp_img)
                        else:
                            total_rest_pred0[rest_cnt%60,:] = pred_2d
                            total_rest_pred1[rest_cnt%60,:] = pred_2d
                            total_rest_pred4[rest_cnt%60,:] = pred_2d
                            total_rest_pred5[rest_cnt%60,:] = pred_2d
                            
                            if (rest_cnt/30):
                                rest_img1.append(temp_img)
                            else:
                                rest_img2.append(temp_img)
                                
                        rest_cnt += 1
                            
                        
                if i % 30 == 29: # 3D CNN Inference every 30 iterations
                    save_len = len(save_img)
                    
                    if save_len < 16:
                        # What should i do?
                        if save_len < 8: # 영상이 8개 미만
                            pred_3d = torch.zeros(30,8).to(device)
                        else: 
                            for k in range(16-save_len):
                                save_img.append(save_img[k])
                            save_img = torch.unsqueeze(torch.cat(save_img, dim=0), 0).permute(0, 2, 1, 3, 4).to(device)
                            #print(save_img.shape)
                            pred_3d = softmax(cnn_3d(save_img))
                        
                    else: # 영상이 16개 이상
                        save_img = torch.cat(save_img, dim=0)
                        # 16개 추출
                        random.seed(0)
                        random_idx = torch.Tensor(random.sample(range(0,save_len),16)).long()
                        random_idx, _ = torch.sort(random_idx)
                        save_img = torch.unsqueeze(save_img[random_idx], 0).permute(0, 2, 1, 3, 4).to(device)
                        #print(save_img.shape)
                        pred_3d = softmax(cnn_3d(save_img))
                        
                    save_pred0[i%60-29:i%60+1, :] += pred_3d
                    save_pred2[i%60-29:i%60+1, :] += pred_3d
                    save_pred4[i%60-29:i%60+1, :] += pred_3d
                    save_pred6[i%60-29:i%60+1, :] += pred_3d
                    
                    save_img = [] # Initialize
                
                if i % 60 == 59: # Audio CNN Inference every 60 iterations
                    # Make Audio Spectrum
                    jj = int(i / 60)
                    ny = y[sr*(jj-1)*2:sr*(jj+1)*2] # 60
                    melspec_each = spec_to_image(melspec_f(ny))
                    plt.axis('off')
                    plt.imshow(melspec_each.T, aspect='auto', origin='lower')
                    plt.savefig("./audio.png", bbox_inches="tight", pad_inches=0)
                    plt.clf()
                    
                    temp_aud = Image.open("./audio.png").convert('RGB')
                    
                    my_spect = torch.unsqueeze(trans(temp_aud), 0).to(device)
                    pred_aud = softmax(cnn_aud(my_spect))
                    save_pred0 += (pred_aud * args.weight)
                    save_pred3 += (pred_aud * args.weight)
                    save_pred5 += (pred_aud * args.weight)
                    save_pred6 += (pred_aud * args.weight)
                    
                    #save_pred0 /= 3.0
                    #save_pred4 /= 2.0
                    #save_pred5 /= 2.0
                    #save_pred6 /= 2.0
                    
                    for j in range(60):                    
                        with open(save_name0, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred0[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name1, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred1[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name2, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred2[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name3, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred3[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name4, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred4[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name5, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred5[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        with open(save_name6, "a") as fd: # txt 생성
                            _, my_preds = torch.max(save_pred6[j,:], 0)
                            fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                        
                    save_pred0 = torch.zeros(60,8).to(device)
                    save_pred1 = torch.zeros(60,8).to(device)
                    save_pred2 = torch.zeros(60,8).to(device)
                    save_pred3 = torch.zeros(60,8).to(device)
                    save_pred4 = torch.zeros(60,8).to(device)
                    save_pred5 = torch.zeros(60,8).to(device)
                    save_pred6 = torch.zeros(60,8).to(device)
                    
            # Rest Image Prediction
            if nam1: 
                y_ = y[::-1]
                ny_ = y_[sr*(0-1)*2:sr*(0+1)*2] # 60
                ny = ny_[::-1]
                melspec_each = spec_to_image(melspec_f(ny))
                plt.axis('off')
                plt.imshow(melspec_each.T, aspect='auto', origin='lower')
                plt.savefig("./audio.png", bbox_inches="tight", pad_inches=0)
                plt.clf()
                
                temp_aud = Image.open("./audio.png").convert('RGB')
                my_spect = torch.unsqueeze(trans(temp_aud), 0).to(device)
                pred_aud = softmax(cnn_aud(my_spect))
                 
                if flag:               
                    save_len = len(rest_img1)    
                    if save_len < 16:
                        # What should i do?
                        if save_len < 8: # 영상이 8개 미만
                            pred_3d = torch.zeros(30,8).to(device)
                        else: 
                            for k in range(16-save_len):
                                rest_img1.append(rest_img1[k])
                            rest_img1 = torch.unsqueeze(torch.cat(rest_img1, dim=0), 0).permute(0, 2, 1, 3, 4).to(device)
                            pred_3d = softmax(cnn_3d(rest_img1))
                        
                    else: # 영상이 16개 이상
                        rest_img1 = torch.cat(rest_img1, dim=0)
                        # 16개 추출
                        random.seed(0)
                        random_idx = torch.Tensor(random.sample(range(0,save_len),16)).long()
                        random_idx, _ = torch.sort(random_idx)
                        rest_img1 = torch.unsqueeze(rest_img1[random_idx], 0).permute(0, 2, 1, 3, 4).to(device)
                        pred_3d = softmax(cnn_3d(rest_img1))
                        
                    total_rest_pred0 += pred_3d
                    total_rest_pred2 += pred_3d
                    total_rest_pred4 += pred_3d
                    total_rest_pred6 += pred_3d
                    
                    total_rest_pred0 = total_rest_pred0[30-nam1:30,:]
                    total_rest_pred2 = total_rest_pred2[30-nam1:30,:]
                    total_rest_pred4 = total_rest_pred4[30-nam1:30,:]
                    total_rest_pred6 = total_rest_pred6[30-nam1:30,:]
                    
                else:
                    for q in range(2):
                        if q:
                            rest_img = rest_img2
                        else:
                            rest_img = rest_img1
                            
                        save_len = len(rest_img)    
                        
                        if save_len < 16:
                            # What should i do?
                            if save_len < 8: # 영상이 8개 미만
                                pred_3d = torch.zeros(30,8).to(device)
                            else: 
                                for k in range(16-save_len):
                                    rest_img.append(rest_img[k])
                                rest_img = torch.unsqueeze(torch.cat(rest_img, dim=0), 0).permute(0, 2, 1, 3, 4).to(device)
                                pred_3d = softmax(cnn_3d(rest_img))
                            
                        else: # 영상이 16개 이상
                            rest_img = torch.cat(rest_img, dim=0)
                            # 16개 추출
                            random.seed(0)
                            random_idx = torch.Tensor(random.sample(range(0,save_len),16)).long()
                            random_idx, _ = torch.sort(random_idx)
                            rest_img = torch.unsqueeze(rest_img[random_idx], 0).permute(0, 2, 1, 3, 4).to(device)
                            pred_3d = softmax(cnn_3d(rest_img))
                            
                        if q:
                            total_rest_pred0[30:60,:] += pred_3d
                            total_rest_pred2[30:60,:] += pred_3d
                            total_rest_pred4[30:60,:] += pred_3d
                            total_rest_pred6[30:60,:] += pred_3d
                        else:
                            total_rest_pred0[0:30,:] += pred_3d
                            total_rest_pred2[0:30,:] += pred_3d
                            total_rest_pred4[0:30,:] += pred_3d
                            total_rest_pred6[0:30,:] += pred_3d
                            
                    total_rest_pred0 = total_rest_pred0[60-nam1:60,:]
                    total_rest_pred2 = total_rest_pred2[60-nam1:60,:]
                    total_rest_pred4 = total_rest_pred4[60-nam1:60,:]
                    total_rest_pred6 = total_rest_pred6[60-nam1:60,:]
                    
                
                total_rest_pred0 += (pred_aud * args.weight)
                total_rest_pred3 += (pred_aud * args.weight)
                total_rest_pred5 += (pred_aud * args.weight)
                total_rest_pred6 += (pred_aud * args.weight)
                
                #total_rest_pred0 /= 3.0
                #total_rest_pred4 /= 2.0
                #total_rest_pred5 /= 2.0
                #total_rest_pred6 /= 2.0
                
                for k in range(total_rest_pred0.shape[0]):                    
                    with open(save_name0, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred0[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name1, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred1[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name2, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred2[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name3, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred3[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name4, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred4[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name5, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred5[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')   
                    with open(save_name6, "a") as fd: # txt 생성
                        _, my_preds = torch.max(total_rest_pred6[k,:], 0)
                        fd.write('%d' % (GT_list[int(my_preds)]) + '\n')
                    
    print('done.')