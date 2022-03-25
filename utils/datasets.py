from torchvision import datasets
import os
import torch

import numpy as np
import cv2
import natsort as nt
import shutil
from random import *
from torch.utils.data import Dataset
from torchvision import transforms
import torch

def LoadDataset(args, data_dir, data_transforms):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bs,
                                                 shuffle=True, num_workers=args.nw)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names
    
class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, args, dataset='crimes', split='train', clip_len=16, preprocess=True, tm=0):



        #version = 1
        print(args.dataset, split)

        self.resize_height = 256
        self.resize_width = 256



        path = os.path.join(os.getcwd(), 'dataset/aff_wild_video')
        if split =='train':
            path_target = os.path.join(path, 'train')
        elif split == 'val':
            path_target = os.path.join(path, 'val')
        else:
            path_target = os.path.join(path, 'test')
 
    
        self.output_dir = path
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1

        self.crop_size_299 = 299
        self.crop_size_224 = args.img
        self.tm = 0
        if split == 'train':
            self.tm_new = 0
        else:
            self.tm_new = 1
        self.fnames, labels = [], []
        for label in nt.natsorted(os.listdir(folder)):
            for fname in nt.natsorted(os.listdir(os.path.join(folder, label))):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(nt.natsorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in nt.natsorted(labels)], dtype=int)




    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        #print(index)
        #print(self.fnames[index])
        buffer = self.load_frames(self.fnames[index], self.tm_new)
        """
        if self.test_m == 0:
            if self.sel_network == 'xception':

                buffer = self.crop(buffer, self.clip_len, self.crop_size_299)
            #else:
            #    #buffer = self.crop(buffer, self.clip_len, self.crop_size_224)
        """
        labels = np.array(self.label_array[index])
        """
        if self.test_m== 0:
            print("af")
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        """
        #print(buffer)
        #buffer = self.normalize(buffer)
        #buffer = self.to_tensor(buffer)
        #print(buffer)

        return buffer, torch.from_numpy(labels)#torch.from_numpy(buffer), torch.from_numpy(labels)


    def preprocess(self):
        if not os.path.exists(self.output_dir):
            pass
            #os.mkdir(self.output_dir)
            #os.mkdir(os.path.join(self.output_dir, 'train'))
            #os.mkdir(os.path.join(self.output_dir, 'val'))
            #os.mkdir(os.path.join(self.output_dir, 'test'))
        """
        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')
    """
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            #frame -= np.array([[[90.0, 98.0, 102.0]]])
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir, tm_new):

        """ 3mode to load 16 frames

        1. uniform sampling 
        2. None-uniform sampling - load 16frame from random number
        3. None-uniform sampling - load 16frame from random number but front, middle, end

        """
        """
        if self.sel_network =='mobilenet-v2' or self.sel_network == 'vgg16' or self.sel_network == 'res101' or self.sel_network == 'shufflenet-v2' or self.sel_network =='mnas':
            self.resize_height = 224
            self.resize_width = 224
        elif sel_network =='xception':
            self.resize_height = 299
            self.resize_width = 299
        """
        
        transform_video_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
           ])
        transform_video_val = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
           ])
        frames = nt.natsorted([os.path.join(file_dir, img) for img in nt.natsorted(os.listdir(file_dir))])
        #print(len(frames))
        if len(frames) < 16:
            print(file_dir)
        frame_count = len(frames) # ex 153 -> 16
        #print(frame_count)
        #buffer = np.empty((16, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        frame_count_div = int(frame_count / 3)
        frame_count_q = 1#int(frame_count / 16)
        num_list = [x for x in range(frame_count)]
        #print(num_list)
        #mode_selection = randint(1,3)   
        mode_selection = 1
        buffer = torch.FloatTensor(3,16,224,224)
        #print(frame_count, mode_selection)

        if mode_selection == 1:
            real_list = []            
            count = 0            
            #print("frame length", len(frames))
            r_crop = randint(0,32)
            for i, frame_name in enumerate(frames):
                #print(i, frame_count_q)

                frame = cv2.imread(frame_name, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if tm_new == 0: # train        
                    frame = cv2.resize(frame, (self.resize_height,self.resize_width), interpolation=cv2.INTER_LINEAR)
                    #print(frame.shape)
                    frame = frame[r_crop:r_crop+224,r_crop:r_crop+224,:] 
                elif tm_new == 1:
                    frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_LINEAR)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2,0,1)
                #frame = np.array(frame).astype(np.uint8)#(np.float64)
                buffer[:,count,:,:] = frame.float()
                #print("count", count)
                count += 1
                if count ==16:
                    break
            #print("final count", count)
        if tm_new == 0:
            for iii in range(16):
        
                frame_each = buffer[:,iii,:,:]
                frame_each = transform_video_train(frame_each)
            
                buffer[:,iii,:,:,] = frame_each
        else:
            for iii in range(16):
        
                frame_each = buffer[:,iii,:,:]
                frame_each = transform_video_val(frame_each)
            
                buffer[:,iii,:,:,] = frame_each

            
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        #print("Crop", buffer.shape)
        if (buffer.shape[0]<=16):
            time_index=0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        
        #buffer = buffer[:clip_len,16:16+crop_size, 16:16+crop_size, :]
        return buffer
