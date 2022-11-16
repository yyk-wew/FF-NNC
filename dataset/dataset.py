import os
import cv2
import numpy as np
import torch
import random
from torch.utils import data
from torchvision import transforms as trans
from PIL import Image

class DataCollector(data.Dataset):
    def __init__(self, path, size=None, label=None, isTrain=False, drop_rate=0):
        self._path = path
        self._image_list = list()
        self._label = np.int64(label)
        self.size = size
        self.isTrain = isTrain

        # transform
        if self.isTrain:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(p=0.5),
                trans.ToTensor(),])
        else:
            self.transform = trans.Compose([
                trans.ToTensor(),])

        # walk through folder
        for dirpath, dirnames, filenames in os.walk(path):
            temp_list = [os.path.join(dirpath, file_name) for file_name in filenames]
            self._image_list = self._image_list + temp_list

        # drop
        if drop_rate != 0:
            new_len = int((1-drop_rate) * len(self._image_list))
            random.shuffle(self._image_list)
            self._image_list = self._image_list[:new_len]

        assert self._label != None

    def __len__(self):
        return len(self._image_list)
        
    def __getitem__(self, index):
        image_path = self._image_list[index]
        img = Image.open(image_path)
        img = img.resize((self.size, self.size))
        img = self.transform(img)
        img = img * 2. - 1.
        
        label = self._label
        return img, label


def DatasetWrapper(label_path_dict, **kwargs):
    # label_path_dict -- {0:[real_path_1, real_path_2, ...], 1:[fake_path_1, fake_path_2, ...]}
    all_dst = list()
    for label, path_list in label_path_dict.items():
        dset_list = [DataCollector(path=p, label=label, **kwargs) for p in path_list]
        all_dst.extend(dset_list)
    return torch.utils.data.ConcatDataset(all_dst)


def get_test_dataset(dataset_name, img_size=256):
    name_book = {
        "celebdf" : {
            0.:["/4T/yike/celebdf/real/"], 
            1.:["/4T/yike/celebdf/fake/"]
        }, 
        "deeperforensic" : {
            1.:["/4T/shengming/DeepfakesDetection_ijcai/dataset/DeeperForensics/manipulated_videos/end_to_end/"]
        }
    }

    # assert
    dataset_name = dataset_name.lower()
    if dataset_name not in name_book.keys():
        raise RuntimeError("Invalid dataset name for evaluation.")

    # construct dataset
    label_path_dict = name_book[dataset_name]
    dataset = DatasetWrapper(label_path_dict, size=img_size, isTrain=False)

    return dataset
    

def get_FF_dataset(dataset_path, mode=None, category_list=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'], \
    img_size=256, isTrain=True, drop_rate=0):
    
    assert mode in ['train', 'valid', 'test']

    path_book = {}
    
    # real (oversampling)
    if isTrain:
        path_book[0.] = [os.path.join(dataset_path, mode, 'real') for i in range(len(category_list))]
    else:
        path_book[0.] = [os.path.join(dataset_path, mode, 'real')]
    
    # fake
    path_for_fake = os.path.join(dataset_path, mode, 'fake')
    path_book[1.] = [os.path.join(path_for_fake, category) for category in category_list]

    dataset = DatasetWrapper(path_book, size=img_size, isTrain=True, drop_rate=drop_rate)
    return dataset
    

def get_FF_5_class(dataset_path, mode=None, category_list=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'], \
    img_size=256, isTrain=True, drop_rate=0):

    assert mode in ['train', 'valid', 'test']

    path_book = {}

    path_book[0.] = [os.path.join(dataset_path, mode, 'real')]
    path_for_fake = os.path.join(dataset_path, mode, 'fake')
    for i in range(len(category_list)):
        path_book[float(i+1)] = [os.path.join(path_for_fake, category_list[i])]
    
    dataset = DatasetWrapper(path_book, size=img_size, isTrain=True, drop_rate=drop_rate)
    return dataset