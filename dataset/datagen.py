from __future__ import absolute_import, division

import os
import sys
import numpy as np
import glob
import random

from PIL import Image
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from options import *


class Pair(data.Dataset):
    def __init__(self, root_dir, subset='train', transform=None, config=None, pairs_per_video=25,
                 stats_path='?.mat', frame_range=100, rand_choice=True):
        super(Pair, self).__init__()
        assert subset in ['train', 'val']
        self.root_dir = root_dir
        self.subset = subset
        self.seq_names = []
        self.anno_dirs = []

        if self.subset == 'train':
            self.seq_dirs = sorted(
                glob.glob(os.path.join(self.root_dir, '???')))
            for s in self.seq_dirs:
                s_splited = s.lstrip().rstrip().split('/')
                self.seq_names.append(os.path.basename(s))
                self.anno_dirs.append(os.path.join(
                    self.root_dir, '???', *s_splited[-2:]))
        elif self.subset == 'val':
            self.seq_dirs = sorted(
                glob.glob(os.path.join(self.root_dir, '???')))
            for s in self.seq_dirs:
                self.seq_names.append(os.path.basename(s))
                self.anno_dirs.append(os.path.join(
                    self.root_dir, '???', self.seq_names[-1]))

        self.transform = transform
        self.stats = load_stats(stats_path)
        self.pairs_per_video = pairs_per_video
        self.frame_range = frame_range
        self.rand_choice = rand_choice

        n = len(self.seq_names)
        self.indices = np.arange(0, n, dtype=int)
        self.indices = np.tile(self.indices, self.pairs_per_video)

        self.exemplarSize = config.exemplarSize
        self.instanceSize = config.instanceSize
        self.scoreSize = config.scoreSize
        self.context = config.context
        self.rPos = config.rPos
        self.rNeg = config.rNeg
        self.totalStride = config.totalStride
        self.ignoreLabel = config.ignoreLabel

    def __getitem__(self, index):

        if self.rand_choice:
            index = np.random.choice(self.indices)

        anno_files = sorted(
            glob.glob(os.path.join(self.anno_dirs[index], '*.xml')))
        objects = [ET.ElementTree(file=f).findall('object')
                   for f in anno_files]
        track_ids, counts = np.unique([obj.find(
            'trackid').text for group in objects for obj in group], return_counts=True)
        track_id = random.choice(track_ids[counts >= 2])

        frames = []
        anno = []

        for f, group in enumerate(objects):
            for obj in group:
                if not obj.find('trackid').text == track_id:
                    continue
                frames.append(f)
                anno.append([
                    int(obj.find('bndbox/xmin').text),
                    int(obj.find('bndbox/ymin').text),
                    int(obj.find('bndbox/xmax').text),
                    int(obj.find('bndbox/ymax').text)
                ])
        img_files = [os.path.join(
            self.seq_dirs[index], '%06d.JPEG' % f) for f in frames]
        anno = np.array(anno)
        anno[:, 2:] = anno[:, 2:] - anno[:, :2] + 1

        rand_z, rand_x = self.sample_pair(len(img_files))
        img_z = Image.open(img_files[rand_z])
        img_x = Image.open(img_files[rand_x])
        if img_z.mode == 'L':
            img_z = img_z.convert('RGB')
            img_x = img_x.convert('RGB')

        bndbox_z = anno[rand_z, :]
        bndbox_x = anno[rand_x, :]
        crop_z = self.crop(img_z, bndbox_z, self.exemplarSize)
        crop_x = self.crop(img_x, bndbox_x, self.instanceSize)

        labels, weights = self.create_labels()
        labels = torch.from_numpy(labels).float()
        weights = torch.from_numpy(weights).float()

        crop_z = self.transform(crop_z)*255.0
        crop_x = self.transform(crop_x)*255.0

        if self.subset == 'train':
            offset_z = np.reshape(
                np.dot(self.stats.rgb_variance_z, np.random.randn(3, 1)), (3, 1, 1))
            offset_x = np.reshape(
                np.dot(self.stats.rgb_variance_x, np.random.randn(3, 1)), (3, 1, 1))
            crop_z += torch.from_numpy(offset_z).float()
            crop_x += torch.from_numpy(offset_x).float()
            crop_z = torch.clamp(crop_z, 0.0, 255.0)
            crop_x = torch.clamp(crop_x, 0.0, 255.0)

        return crop_z, crop_x, labels, weights

    def __len__(self):
        return len(self.indices)

    def sample_pair(self, n):
        rand_z = np.random.randint(n)
        possible_x = np.arange(rand_z-self.frame_range,
                               rand_z+self.frame_range)
        possible_x = np.intersect1d(possible_x, np.arange(n))
        possible_x = possible_x[possible_x != rand_z]
        rand_x = np.random.choice(possible_x)
        return rand_z, rand_x

    def crop(self, image, bndbox, out_size):
        center = bndbox[:2]+bndbox[2:]/2
        size = bndbox[2:]

        context = self.context * size.sum()
        patch_sz = out_size/self.exemplarSize*np.sqrt((size+context).prod())

        return crop_pil(image, center, patch_sz, out_size=out_size)

    def create_labels(self):
        labels = self.create_logisticloss_labels()
        weights = np.zeros_like(labels)

        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights[labels == 1] = 0.5/pos_num
        weights[labels == 0] = 0.5/neg_num

        labels = labels[np.newaxis, :]
        weights = weights[np.newaxis, :]

        return labels, weights

    def create_logisticloss_labels(self):
        label_sz = self.scoreSize
        r_pos = self.rPos/self.totalStride
        r_neg = self.rNeg/self.totalStride
        labels = np.zeros((label_sz, label_sz))

        for r in range(label_sz):
            for c in range(label_sz):
                dist = np.sqrt((r-label_sz//2)**2+(c-label_sz//2)**2)
                if dist <= r_pos:
                    labels[r, c] = 1
                elif dist <= r_neg:
                    labels[r, c] = self.ignoreLabel
                else:
                    labels[r, c] = 0

        return labels

class train_config(object):
    exemplarSize=127
    instanceSize=255
    scoreSize=17
    context=0.5
    rPos=16
    rNeg=0
    totalStride=8
    ignoreLabel=-100

if __name__=='__main__':
    config = train_config()
    transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    pair_train=Pair(root_dir='???',subset='train',transform=transforms_train,config=config)
    print(pair_train.__len__())
    pair_train.__getitem__(20)?

    transforms_val = transforms.ToTensor()
    pair_val=Pair(root_dir='???',subset='val',transform=transforms_val,config=config)
    print(pair_val.__len__())
    pair_val.__getitem__(20)?

    

class Samples(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts):

        self.img_list = np.array([os.path.join(img_dir, img)
                                  for img in img_list])
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator(
            'gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator(
            'uniform', image.size, 1, 1.2, 1.1, True)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames,
                           len(self.img_list))
        idx = self.index[self.pointer:next_pointer]

        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            n_pos = (self.batch_pos - len(pos_regions)
                     ) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions)
                     ) // (self.batch_frames - i)
            pos_examples = gen_samples(
                self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(
                self.neg_generator, bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions = np.concatenate(
                (pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate(
                (neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions).float()
        neg_regions = torch.from_numpy(neg_regions).float()
        return pos_regions, neg_regions
    next = __next__

    def extract_regions(slice, image, samples):
        regions = np.zeros((len(samples), self.crop_size,
                            self.crop_size, 3), dtype='unit8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(
                image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
