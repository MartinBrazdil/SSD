import csv
import sys

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from detection_tutorial.utils import transform
import numpy as np


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


# class UNITDataset_old(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#
#     def __init__(self, data_folder, split, keep_difficult=False):
#         """
#         :param data_folder: folder where data files are stored
#         :param split: split, one of 'TRAIN' or 'TEST'
#         :param keep_difficult: keep or discard objects that are considered difficult to detect?
#         """
#         self.split = split.upper()
#
#         assert self.split in {'TRAIN', 'TEST'}
#
#         self.data_folder = data_folder
#         self.keep_difficult = keep_difficult
#
#         default_ctgs = {'bracelet': 0,
#                         'figure': 1,
#                         'gangsta_car': 2,
#                         'keys': 3,
#                         'plate': 4,
#                         'sports_car': 5}
#
#         self.annotations, self.images = [], []
#         for ctg_dir in os.scandir(data_folder):
#             if ctg_dir.is_dir():
#                 for file in os.listdir(ctg_dir):
#                     if file.endswith('.csv'):
#                         with open(os.path.join(ctg_dir.path, file), 'rt', newline='\n') as ann_csv:
#                             csv_ann = csv.reader(ann_csv)
#                             for ann_row in csv_ann:
#                                 ctg, x_min, y_min, x_max, y_max = ann_row[0].split(' ')
#                                 ctg_idx = default_ctgs[ctg]
#                                 ann = {'boxes': [[int(x_min), int(y_min), int(x_max), int(y_max)]],
#                                        'labels': [[ctg_idx]], 'difficulties': [False]}
#                                 self.annotations.append(ann)
#                     if file.endswith('.jpg'):
#                         img = Image.open(os.path.join(ctg_dir.path, file))
#                         # img = img.resize([img.size[0] // 5, img.size[1] // 5], Image.BICUBIC)
#                         # with open(os.path.join(ctg_dir.path, file), 'wb') as img_file:
#                         #     img.save(img_file)
#                         self.images.append(img)
#
#         assert len(self.images) == len(self.annotations)
#
#     def __getitem__(self, i):
#         boxes = torch.FloatTensor(self.annotations[i]['boxes'])  # (n_objects, 4)
#         labels = torch.LongTensor(self.annotations[i]['labels'])  # (n_objects)
#         difficulties = torch.ByteTensor(self.annotations[i]['difficulties'])  # (n_objects)
#
#         # Apply transformations
#         image, boxes, labels, difficulties = transform(self.images[i], boxes, labels, difficulties, split=self.split)
#
#         return image, boxes, labels, difficulties
#
#     def __len__(self):
#         return len(self.images)
#
#     def collate_fn(self, batch):
#         """
#         Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
#
#         This describes how to combine these tensors of different sizes. We use lists.
#
#         Note: this need not be defined in this Class, can be standalone.
#
#         :param batch: an iterable of N sets from __getitem__()
#         :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
#         """
#
#         images = list()
#         boxes = list()
#         labels = list()
#         difficulties = list()
#
#         for b in batch:
#             images.append(b[0])
#             boxes.append(b[1])
#             labels.append(b[2])
#             difficulties.append(b[3])
#
#         images = torch.stack(images, dim=0)
#
#         return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class UNITDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    # # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.buffer =[]
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        self.label_color_map = {}
        self.label_map, self.rev_label_map = {}, {}
        self.annotations, self.images = [], []
        dirs = os.listdir(data_folder)
        last_dir = str(sorted([int(d) for d in dirs])[-1])
        with open(os.path.join(data_folder, last_dir, 'image.jpg'), 'rb') as imf:
            img = Image.open(imf)
            img.load()
            self.images.append(img)
        with open(os.path.join(data_folder, last_dir, 'anns.json')) as af:
            anns = json.load(af)
            img_w, img_h = img.size
            self.n_classes = len(anns)+1
            bboxes, labels, difficulties = [], [], []
            for ctg_idx, (ctg, (x, y, w, h)) in enumerate(anns.items()):
                self.label_map[ctg] = ctg_idx + 1
                self.rev_label_map[ctg_idx + 1] = ctg
                self.label_color_map[ctg] = self.distinct_colors[ctg_idx]
                bboxes.append([x, y, x+w, y+h])
                labels.append(ctg_idx + 1)
                difficulties.append(False)
            ann = {'boxes': bboxes, 'labels': labels, 'difficulties': difficulties}
            self.annotations.append(ann)
            self.label_map['background'] = 0

    def __getitem__(self, i):
        if self.split == 'TRAIN':
            i = i % len(self.annotations)
        boxes = torch.FloatTensor(self.annotations[i]['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(self.annotations[i]['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(self.annotations[i]['difficulties'])  # (n_objects)

        # Apply transformations
        if np.random.binomial(1, 0.8) and 0 < len(self.buffer):
            image, boxes, labels, difficulties = self.buffer[np.random.uniform(0, len(self.bufffer)-1)]
        else:
            image, boxes, labels, difficulties = transform(self.images[i], boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        if self.split == 'TRAIN':
            return 1000*len(self.images)
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


if __name__ == '__main__':
    dataset = UNITDataset('/host_home/projects/data/unit/', 'TRAIN')
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))
