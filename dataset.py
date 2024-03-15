import os
import pdb

import cv2
import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

filepath = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'


# def default_loader(path,grey=0):
def default_loader(path):
    # return Image.open(path)
    # pdb.set_trace()
    return Image.open(path).convert('RGB')
    # pdb.set_trace()
    # if grey==0:
    #     img = cv2.imread(path)
    #     img = cv2.resize(img, (384, 384))
    #     img = img / 255.0
    #     img = torch.Tensor(img)
    # else:
    #      img = cv2.imread(path, 0)
    #      img = cv2.resize(img, (384, 384))
    #      img = torch.Tensor(img)

    # return img


def default_loader_label(path):
    # return Image.open(path)
    # pdb.set_trace()
    return Image.open(path).convert('L')


class Busi(Dataset):
    def __init__(self, imagefilepath, imagefilepath_label, transform, transform_test, loader=default_loader,
                 loader_label=default_loader_label):
        imgs = []
        labels = []
        total_img = os.listdir(imagefilepath)
        total_label = os.listdir(imagefilepath_label)
        for img in total_img:
            imgs.append(img)
        for label in total_label:
            labels.append(label)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform_test = transform_test
        self.loader = loader
        self.loader_label = loader_label
        self.imagefilepath = imagefilepath
        self.imagefilepath_label = imagefilepath_label

    def __getitem__(self, index):
        imageSize = len(self.imgs)
        # fn, label = self.imgs[index]
        img_name = self.imgs[index]
        label_name = self.labels[index]
        # pdb.set_trace()
        img = self.loader(self.imagefilepath + img_name)
        label = self.loader_label(self.imagefilepath_label + label_name)
        # label = self.loader(self.imagefilepath_label + label_name,grey=1)
        img = self.transform(img)
        label = self.transform_test(label)

        return img, label

    def name(self):
        return self.imgs[0]

    def __len__(self):
        return len(self.imgs)


class K_fold1(Dataset):
    def __init__(self, img_list, label_list, imagefilepath, imagefilepath_label, transform, transform_test,
                 loader=default_loader, loader_label=default_loader_label):
        imgs = []
        labels = []
        total_img = img_list
        total_label = label_list
        for img in total_img:
            imgs.append(img)
        for label in total_label:
            labels.append(label)
        # self.image_paths = sorted(
        #     [os.path.join(imagefilepath, 'images', f) for f in os.listdir(os.path.join(imagefilepath, 'images'))])
        # self.mask_paths = sorted(
        #     [os.path.join(imagefilepath_label, 'masks', f) for f in os.listdir(os.path.join(imagefilepath_label, 'masks'))])

        # pdb.set_trace()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform_test = transform_test
        self.loader = loader
        self.loader_label = loader_label
        self.imagefilepath = imagefilepath
        self.imagefilepath_label = imagefilepath_label

    def __getitem__(self, index):
        # pdb.set_trace()
        imageSize = len(self.imgs)
        # img_name = self.imgs[0][index]
        # label_name = self.labels[0][index]
        img_name = self.imgs[index]
        label_name = self.labels[index]
        # img = Image.open(self.image_paths[index]).convert('RGB')
        # label = Image.open(self.mask_paths[index]).convert('L')

        img = self.loader(self.imagefilepath + img_name)
        label = self.loader_label(self.imagefilepath_label + label_name)
        # for i in range(len(img_name)):
        #     img_path = self.imagefilepath + img_name[i]
        #     label_path = self.imagefilepath_label + label_name[i]
        # img = self.loader(img_path)
        # label = self.loader(label_path)

        # img = self.loader(self.imagefilepath + img_name)
        # label = self.loader_label(self.imagefilepath_label + label_name)
        # label = self.loader(self.imagefilepath_label + label_name,grey=1)

        img = self.transform(img)
        label = self.transform_test(label)
        return img, label

    def name(self):
        return self.imgs[0]

    def __len__(self):
        return len(self.imgs)


class K_fold_gai(Dataset):
    def __init__(self, data, labels, k, fold_idx, transform=None):
        """
        :param data: 数据
        :param labels: 标签
        :param k: K折交叉验证的K值
        :param fold_idx: 当前折的下标，从0开始
        :param transform: 数据转换操作
        """
        assert k > 1, "K必须大于1"
        assert 0 <= fold_idx < k, "fold_idx必须在[0, k)范围内"
        self.data = data
        self.labels = labels
        self.transform = transform
        self.k = k
        self.fold_idx = fold_idx
        self.idx = np.arange(len(data))  # 数据下标
        self.fold_size = len(data) // k  # 每一折数据大小
        self.start_idx = fold_idx * self.fold_size  # 起始下标
        self.end_idx = self.start_idx + self.fold_size if fold_idx != k - 1 else len(data)  # 结束下标

    def __getitem__(self, index):
        idx = self.idx[self.start_idx:self.end_idx]
        idx = np.delete(idx, index)  # 把验证集的数据删除，得到训练集的下标
        data = self.data[idx]
        labels = self.labels[idx]

        # 数据转换
        if self.transform is not None:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return self.fold_size


class K_fold(Dataset):
    def __init__(self, imagefilepath, imagefilepath_label, transform, transform_test, index=None,
                 loader=default_loader, loader_label=default_loader_label):
        self.imagefilepath = imagefilepath
        self.imagefilepath_label = imagefilepath_label
        self.transform = transform
        self.transform_test = transform_test

        self.imgs = [sorted(os.listdir(imagefilepath))[i] for i in index]
        # self.imgs = sorted(os.listdir(imagefilepath))[index]
        self.labels = [sorted(os.listdir(imagefilepath_label))[i] for i in index]

    def __getitem__(self, index):
        img_name = self.imgs[index]
        label_name = self.labels[index]

        img = Image.open(os.path.join(self.imagefilepath, img_name)).convert('RGB')
        label = Image.open(os.path.join(self.imagefilepath_label, label_name)).convert('L')

        img = self.transform(img)
        label = self.transform_test(label)
        return img, label

    def name(self):
        return self.imgs[0]

    def __len__(self):
        return len(self.imgs)
