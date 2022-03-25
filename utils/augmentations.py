from torchvision import transforms
import numpy as np
import torch
import random

def augmentation_set(args):

    if args.aug == 0:

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif args.aug==1:

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

            ]),
            'val': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }            
    elif args.aug==2:

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(),
            ]),
            'val': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif args.aug==3:

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                
            ]),
            'val': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    return data_transforms
    
def halfmix(args, inputs, labels, rand_index, phase, labels_new_list):
    inputs_temp = inputs.clone()
    labels_temp = labels.clone()
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    labels_list_a = torch.zeros_like(labels).to(device)
    labels_list_b = torch.zeros_like(labels).to(device)


    for iii in range(inputs.size()[0]):
    
        inputs_each = inputs[iii]

        labels_each = labels_temp[iii]   
        r = np.random.rand(1)
        if phase == 'val':
        
            r = 1.0
        
        if r < args.mix_prob:

            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)
                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each #

            label_b = labels_temp[rand_index_each]


            # 원본 왼쪽 또는 위, 복사는 오른쪽 또는 아래
            
            random_hv = np.random.rand(1)
            inputs_par = torch.zeros_like(inputs_each).to(device)
            
            par_c,par_h, par_w = inputs_par.shape

            if random_hv < 0.5 : # 왼오
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    
                    inputs_par[:,:,0:int(par_w*0.5)] = inputs_each[:,:,0:int(par_w*0.5)]
                    inputs_par[:,:,int(par_w*0.5):] = inputs_each_b[:,:,int(par_w*0.5):]
                else:

                    inputs_par[:,:,int(par_w*0.5):] = inputs_each[:,:,int(par_w*0.5):]
                    inputs_par[:,:,0:int(par_w*0.5):] = inputs_each_b[:,:,0:int(par_w*0.5)]
            else : # 위 아래
                random_lr = np.random.rand(1)
                if random_lr < 0.5:

                    inputs_par[:,0:int(par_h*0.5),:] = inputs_each[:,0:int(par_h*0.5),:]
                    inputs_par[:,int(par_h*0.5):,:] = inputs_each_b[:,int(par_h*0.5):,:]
                else:
                    inputs_par[:,int(par_h*0.5):,:] = inputs_each[:,int(par_h*0.5):,:]
                    inputs_par[:,0:int(par_h*0.5),:] = inputs_each_b[:,0:int(par_h*0.5),:]
                    
                
            inputs_temp[iii] = inputs_par

            labels_list_a[iii] = label_a
            labels_list_b[iii] = label_b
        else:
            inputs_temp[iii] = inputs_each
            labels_list_a[iii] = labels_each 
            labels_list_b[iii] = labels_each

    return inputs_temp, labels_list_a, labels_list_b

def halfout(args, inputs, labels, rand_index, phase, labels_new_list):
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    inputs_temp = inputs.clone()
    labels_temp = labels.clone()
    if phase == 'val':
    
        return inputs_temp
    for iii in range(inputs.size()[0]):
        inputs_each = inputs[iii]
        labels_each = labels_temp[iii]
        r = np.random.rand(1)

        
        if r < args.mix_prob:# and epoch < 30:
            
            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)
                #print(r_idx)
                #print(rand_index)
                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #
            #print("each", labels_each) #
            #rand_index_each = rand_index[iii] #
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each #
            #print("each n", label_a) #
            label_b = labels_temp[rand_index_each]

            # 원본 왼쪽 또는 위, 복사는 오른쪽 또는 아래
            
            random_hv = np.random.rand(1)
            inputs_par = torch.zeros_like(inputs_each).to(device)
            
            par_c,par_h, par_w = inputs_par.shape

            torch_zero = torch.Tensor([0]).to(device)
            if random_hv < 0.5 : # 왼오
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    
                    inputs_par[:,:,0:int(par_w*0.5)] = inputs_each[:,:,0:int(par_w*0.5)]
                    inputs_par[:,:,int(par_w*0.5):] = torch_zero
                else:

                    inputs_par[:,:,0:int(par_w*0.5)] = torch_zero
                    inputs_par[:,:,int(par_w*0.5):] = inputs_each[:,:,int(par_w*0.5):]
            else : # 위 아래
                random_lr = np.random.rand(1)
                if random_lr < 0.5:

                    inputs_par[:,0:int(par_h*0.5),:] = inputs_each[:,0:int(par_h*0.5),:]
                    inputs_par[:,int(par_h*0.5):,:] = torch_zero
                else:
                    inputs_par[:,0:int(par_h*0.5),:] = torch_zero
                    inputs_par[:,int(par_h*0.5):,:] = inputs_each[:,int(par_h*0.5):,:]
                
            inputs_temp[iii] = inputs_par

            long_tensor = torch.LongTensor([0.5]).to(device)

            labels[iii] = label_a

        else:
            inputs_temp[iii] = inputs_each

    return inputs_temp
    
def halfmix_jit(args, inputs, labels, rand_index, phase, labels_new_list):

    inputs_temp = inputs.clone()
    labels_temp = labels.clone()
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    labels_list_a = torch.zeros_like(labels).to(device)
    labels_list_b = torch.zeros_like(labels).to(device)

    a_list = [0.4, 0.45, 0.5, 0.55, 0.6]
    b_list = [0.6, 0.55, 0.5, 0.45, 0.4]
    random_jit_r = random.randint(0,len(a_list)-1)
    random_jit = a_list[random_jit_r]
    random_jit2 = b_list[random_jit_r]
    for iii in range(inputs.size()[0]):
                       
        inputs_each = inputs[iii]
        labels_each = labels_temp[iii]   
        r = np.random.rand(1)
        
        if phase == 'val':
        
            r = 1.0

        if r < args.mix_prob:# and epoch < 30:

            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)

                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #


            #rand_index_each = rand_index[iii] #
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each #

            label_b = labels_temp[rand_index_each]

            # 원본 왼쪽 또는 위, 복사는 오른쪽 또는 아래
            
            random_hv = np.random.rand(1)
            inputs_par = torch.zeros_like(inputs_each).to(device)
            
            par_c,par_h, par_w = inputs_par.shape

            if random_hv < 0.5 : # 왼오
                random_lr = np.random.rand(1)


                if random_lr < 0.5:
                    
                    inputs_par[:,:,0:int(par_w*random_jit)] = inputs_each[:,:,0:int(par_w*random_jit)]
                    inputs_par[:,:,int(par_w*random_jit2):] = inputs_each_b[:,:,int(par_w*random_jit2):]
                else:

                    inputs_par[:,:,int(par_w*random_jit):] = inputs_each[:,:,int(par_w*random_jit):]
                    inputs_par[:,:,0:int(par_w*random_jit2):] = inputs_each_b[:,:,0:int(par_w*random_jit2)]
            else : # 위 아래
                random_lr = np.random.rand(1)

                if random_lr < 0.5:

                    inputs_par[:,0:int(par_h*random_jit),:] = inputs_each[:,0:int(par_h*random_jit),:]
                    inputs_par[:,int(par_h*random_jit2):,:] = inputs_each_b[:,int(par_h*random_jit2):,:]
                else:
                    inputs_par[:,int(par_h*random_jit):,:] = inputs_each[:,int(par_h*random_jit):,:]
                    inputs_par[:,0:int(par_h*random_jit2),:] = inputs_each_b[:,0:int(par_h*random_jit2),:]
                    
                
            inputs_temp[iii] = inputs_par
            labels_list_a[iii] = label_a
            labels_list_b[iii] = label_b
        else:
            inputs_temp[iii] = inputs_each
            labels_list_a[iii] = labels_each 
            labels_list_b[iii] = labels_each

    return inputs_temp, labels_list_a, labels_list_b, random_jit,  random_jit2

    
def linejit(args, inputs, labels, rand_index, phase, labels_new_list):

    pass

    return None   
def lineout(args, inputs, labels, rand_index, phase, labels_new_list):

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    inputs_temp = inputs.clone()
    rand_index = torch.randperm(inputs.size()[0]).cuda() # batch 안에서 랜덤하게 뽑음    
    labels_temp = labels.clone()
    
    
    
    for iii in range(inputs.size()[0]):
        inputs_each = inputs[iii]
        labels_each = labels_temp[iii]
        r = np.random.rand(1)
        
        if phase == 'val':
        
            r = 1.0
            
        if r < args.mix_prob:
            
            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)
                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #

            #rand_index_each = rand_index[iii] 
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each 

            label_b = labels_temp[rand_index_each]

            
            random_hv = np.random.rand(1)
            inputs_par = inputs_each.clone()
            
            par_c,par_h, par_w = inputs_par.shape

            torch_zero = torch.Tensor([0]).to(device)
            line_init = random.randint(0,15)
            if random_hv < 0.5 : 
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    
                    inputs_par[:,:,line_init:line_init+10] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
            else : 
                random_lr = np.random.rand(1)
                if random_lr < 0.5:




                    inputs_par[:,line_init:line_init+10,:] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,line_interval + line_init:line_interval +line_init+10,:] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,line_interval + line_init:line_interval +line_init+10,:] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,line_interval + line_init:line_interval +line_init+10,:] = torch_zero
                    line_init = line_init+10
                    line_interval = random.randint(25,35)
                    inputs_par[:,line_interval + line_init:line_interval +line_init+10,:] = torch_zero

                
            inputs_temp[iii] = inputs_par

            long_tensor = torch.LongTensor([0.5]).to(device)

            labels[iii] = label_a

        else:
            inputs_temp[iii] = inputs_each   

    return inputs_temp      
    
def halfmix_out_jit(args, inputs, labels, rand_index, phase, labels_new_list):
    inputs_temp = inputs.clone()
    labels_temp = labels.clone()
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    labels_list_a = torch.zeros_like(labels).to(device)
    labels_list_b = torch.zeros_like(labels).to(device)

    a_list = [0.4, 0.45, 0.5, 0.55, 0.6]
    b_list = [0.6, 0.55, 0.5, 0.45, 0.4]
    random_jit_r = random.randint(0,len(a_list)-1)
    random_jit = a_list[random_jit_r]
    random_jit2 = b_list[random_jit_r]

    for iii in range(inputs.size()[0]):
    
            
        inputs_each = inputs[iii]
        labels_each = labels_temp[iii]   
        r = np.random.rand(1)
        if phase == 'val':
        
            r = 1.0
        if r < args.mix_prob:
            
            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)
                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #
            #rand_index_each = rand_index[iii] #
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each #
 
            label_b = labels_temp[rand_index_each]

            
            random_hv = np.random.rand(1)
            inputs_par = torch.zeros_like(inputs_each).to(device)
            
            par_c,par_h, par_w = inputs_par.shape

            torch_zero = torch.Tensor([0]).to(device)

            if random_hv < 0.5 :
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    inputs_par[:,:,0:int(par_w*random_jit)] = inputs_each[:,:,0:int(par_w*random_jit)]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,:,int(par_w*random_jit2):] = inputs_each_b[:,:,int(par_w*random_jit2):]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,:,int(par_w*random_jit2):] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a
                else:
                    inputs_par[:,:,int(par_w*random_jit):] = inputs_each[:,:,int(par_w*random_jit):]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,:,0:int(par_w*random_jit2):] = inputs_each_b[:,:,0:int(par_w*random_jit2)]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,:,0:int(par_w*random_jit2)] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a

            else : 
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    inputs_par[:,0:int(par_h*random_jit),:] = inputs_each[:,0:int(par_h*random_jit),:]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,int(par_h*random_jit2):,:] = inputs_each_b[:,int(par_h*random_jit2):,:]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,int(par_h*random_jit2):,:] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a
                else:
                    inputs_par[:,int(par_h*random_jit):,:] = inputs_each[:,int(par_h*random_jit):,:]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,0:int(par_h*random_jit2),:] = inputs_each_b[:,0:int(par_h*random_jit2),:]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,0:int(par_h*random_jit2),:] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b


            inputs_temp[iii] = inputs_par

        else:
            inputs_temp[iii] = inputs_each
            labels_list_a[iii] = labels_each 
            labels_list_b[iii] = labels_each

    return inputs_temp, labels_list_a, labels_list_b,random_jit,random_jit2    
def halfmix_out_jit_lineout_notuse(args, inputs, labels, rand_index, phase, labels_new_list):
    inputs_temp = inputs.clone()
    labels_temp = labels.clone()
    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    labels_list_a = torch.zeros_like(labels).to(device)
    labels_list_b = torch.zeros_like(labels).to(device)

    a_list = [0.4, 0.45, 0.5, 0.55, 0.6]
    b_list = [0.6, 0.55, 0.5, 0.45, 0.4]
    random_jit_r = random.randint(0,len(a_list)-1)
    random_jit = a_list[random_jit_r]
    random_jit2 = b_list[random_jit_r]

    for iii in range(inputs.size()[0]):
    
            
        inputs_each = inputs[iii]
        labels_each = labels_temp[iii]   
        r = np.random.rand(1)
        if phase == 'val':
        
            r = 1.0
        if r < args.mix_prob:
            
            if args.imbalance == 1:
            
                if len(labels_new_list) == 0:
                    r_idx = 0
                else:
                    r_idx = random.randint(0,len(labels_new_list)-1)
                rand_index_each = rand_index[r_idx] #
            else:
                rand_index_each = rand_index[iii] #
            #rand_index_each = rand_index[iii] #
            inputs_each_b = inputs[rand_index_each]
            label_a = labels_each #
 
            label_b = labels_temp[rand_index_each]

            
            random_hv = np.random.rand(1)
            inputs_par = torch.zeros_like(inputs_each).to(device)
            
            par_c,par_h, par_w = inputs_par.shape

            torch_zero = torch.Tensor([0]).to(device)
            line_init = random.randint(0,15)
            if random_hv < 0.5 :
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    inputs_par[:,:,0:int(par_w*random_jit)] = inputs_each[:,:,0:int(par_w*random_jit)]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,:,int(par_w*random_jit2):] = inputs_each_b[:,:,int(par_w*random_jit2):]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,:,int(par_w*random_jit2):] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a
                else:
                    inputs_par[:,:,int(par_w*random_jit):] = inputs_each[:,:,int(par_w*random_jit):]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,:,0:int(par_w*random_jit2):] = inputs_each_b[:,:,0:int(par_w*random_jit2)]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,:,0:int(par_w*random_jit2)] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a
                inputs_par[:,:,line_init:line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero 
            else : 
                random_lr = np.random.rand(1)
                if random_lr < 0.5:
                    inputs_par[:,0:int(par_h*random_jit),:] = inputs_each[:,0:int(par_h*random_jit),:]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,int(par_h*random_jit2):,:] = inputs_each_b[:,int(par_h*random_jit2):,:]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,int(par_h*random_jit2):,:] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_a
                else:
                    inputs_par[:,int(par_h*random_jit):,:] = inputs_each[:,int(par_h*random_jit):,:]
                    r_mixout = np.random.rand(1)
                    if r_mixout < 0.5:
                        inputs_par[:,0:int(par_h*random_jit2),:] = inputs_each_b[:,0:int(par_h*random_jit2),:]
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                    else:
                        inputs_par[:,0:int(par_h*random_jit2),:] = torch_zero
                        labels_list_a[iii] = label_a
                        labels_list_b[iii] = label_b
                inputs_par[:,:,line_init:line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero
                line_init = line_init+10
                line_interval = random.randint(25,35)
                inputs_par[:,:,line_interval + line_init:line_interval +line_init+10] = torch_zero 

            inputs_temp[iii] = inputs_par

        else:
            inputs_temp[iii] = inputs_each
            labels_list_a[iii] = labels_each 
            labels_list_b[iii] = labels_each

    return inputs_temp, labels_list_a, labels_list_b,random_jit,random_jit2    
def imbalance_set(args, labels):

    cnt = [0 for i in range(args.nc)] # class 별로 다름
    for idx in labels:
        cnt[idx] += 1

    cnt_sort = sorted(cnt)
    cnt_1 = cnt_sort[0]
    cnt_2 = cnt_sort[1]
    cnt_3 = cnt_sort[2]
    cnt_4 = cnt_sort[3]
    cnt_1_idx = cnt.index(cnt_1)
    cnt_2_idx = cnt.index(cnt_2)
    cnt_3_idx = cnt.index(cnt_3)
    cnt_4_idx = cnt.index(cnt_4)
    
    labels_new_list = []
    
    for p, idx in enumerate(labels):
        #print("here", labels[p], idx)
        if labels[p] == cnt_1_idx or labels[p] == cnt_2_idx or labels[p] == cnt_3_idx or labels[p] == cnt_4_idx:
            labels_new_list.append(p)
    #print("list", labels_new_list)
    if len(labels_new_list) == 0:
    
        labels_new_list.append(p)
    return labels_new_list
    
    
