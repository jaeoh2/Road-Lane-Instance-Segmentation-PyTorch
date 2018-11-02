import torch
from torch.utils import data
from skimage.transform import AffineTransform, warp
from skimage import img_as_float64, img_as_float32, img_as_ubyte
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob
import os


class tuSimpleDataset(data.Dataset):
    # refer from : 
    # https://github.com/vxy10/ImageAugmentation
    # https://github.com/TuSimple/tusimple-benchmark/blob/master/example/lane_demo.ipynb
    def __init__(self, file_path, size=[640, 360], gray=True, train=True, intensity=10):
        warnings.simplefilter("ignore")

        self.width = size[0]
        self.height = size[1]
        self.n_seg = 5
        self.file_path = file_path
        self.flags = {'size':size, 'gray':gray, 'train':train, 'intensity':intensity}
        self.json_lists = glob.glob(os.path.join(self.file_path, '*.json'))
        self.labels = []
        for json_list in self.json_lists:
            self.labels += [json.loads(line) for line in open(json_list)]
        self.lanes = [lane['lanes'] for lane in self.labels]
        self.y_samples = [y_sample['h_samples'] for y_sample in self.labels]
        self.raw_files = [raw_file['raw_file'] for raw_file in self.labels]

        self.img = np.zeros(size, np.uint8)
        self.label_img = np.zeros(size, np.uint8)
        self.ins_img = np.zeros((0,size[0],size[1]), np.uint8)
        
        self.len = len(self.labels)
        
    def random_transform(self):
        intensity=self.flags['intensity']
        def _get_delta(intensity):
            delta = np.radians(intensity)
            rand_delta = np.random.uniform(low=-delta, high=delta)
            return rand_delta

        trans_M = AffineTransform(scale=(.9, .9),
                                 translation=(-_get_delta(intensity), _get_delta(intensity)),
                                 shear=_get_delta(intensity))
        self.img = img_as_float32(self.img)
        self.label_img = img_as_float32(self.label_img)
        self.ins_img = img_as_float32(self.ins_img)

        self.img = warp(self.img, trans_M)
        self.label_img = warp(self.label_img, trans_M)
        for i in range(len(self.ins_img)):
            self.ins_img[i] = warp(self.ins_img[i], trans_M)
    
    def image_resize(self):
        ins = []
        self.img = cv2.resize(self.img, tuple(self.flags['size']), interpolation=cv2.INTER_CUBIC)
        self.label_img = cv2.resize(self.label_img, tuple(self.flags['size']), interpolation=cv2.INTER_CUBIC)
        for i in range(len(self.ins_img)):
            dst = cv2.resize(self.ins_img[i], tuple(self.flags['size']), interpolation=cv2.INTER_CUBIC)
            ins.append(dst)

        self.ins_img = np.array(ins, dtype=np.uint8)
    
    def preprocess(self):
        # CLAHE nomalization
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        img_plane = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_plane[0] = clahe.apply(img_plane[0])
        img = cv2.merge(img_plane)
        self.img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
   
    def get_lane_image(self, idx):
        lane_pts = [[(x,y) for (x,y) in zip(lane, self.y_samples[idx]) if x >= 0] for lane in self.lanes[idx]]
        while len(lane_pts) < self.n_seg:
            lane_pts.append(list())
        self.img = plt.imread(os.path.join(self.file_path, self.raw_files[idx]))
        self.height, self.width, _ = self.img.shape
        self.label_img = np.zeros((self.height, self.width), dtype=np.uint8)
        self.ins_img = np.zeros((0, self.height, self.width), dtype=np.uint8)
        
        for i, lane_pt in enumerate(lane_pts):
            cv2.polylines(self.label_img, np.int32([lane_pt]), isClosed=False, color=(1), thickness=15)
            gt = np.zeros((self.height, self.width), dtype=np.uint8)
            gt = cv2.polylines(gt, np.int32([lane_pt]), isClosed=False, color=(1), thickness=7)
            self.ins_img = np.concatenate([self.ins_img, gt[np.newaxis]])

    def __getitem__(self, idx):
        self.get_lane_image(idx)
        self.image_resize()
        self.preprocess()

        if self.flags['train']:
            #self.random_transform()
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            self.label_img = np.array(self.label_img, dtype=np.float32)
            self.ins_img = np.array(self.ins_img, dtype=np.float32)
            return torch.Tensor(self.img), torch.LongTensor(self.label_img), torch.Tensor(self.ins_img)
        else:
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            return torch.Tensor(self.img)
    
    def __len__(self):
        return self.len
