import torch
from torch.utils import data
from skimage.transform import AffineTransform, warp
from skimage import img_as_float64
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob
import os
import random


class tuSimpleDataset(data.Dataset):
    # refer from : 
    # https://github.com/vxy10/ImageAugmentation
    # https://github.com/TuSimple/tusimple-benchmark/blob/master/example/lane_demo.ipynb
    def __init__(self, file_path, size=[640, 360], gray=True, train=True, intensity=10):
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
        
        self.len = len(self.labels)
        
        
    def warp_affine(self, M):
        self.img = cv2.warpAffine(self.img, M, tuple(self.flags['size']))
        self.label_img = cv2.warpAffine(self.label_img, M, tuple(self.flags['size']))
        
    def random_brightness(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        rand_bright = .15 + np.random.uniform()
        img[:,:,2] = img[:,:,2]*rand_bright
        self.img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
    def random_rotate(self, ang_range=20):
        ang_rot = np.random.uniform(ang_range) - ang_range/2
        Rot_M = cv2.getRotationMatrix2D((self.flags['size'][1]/2, self.flags['size'][0]/2), ang_rot, 1)
        self.warp_affine(Rot_M)
        
    def random_translate(self, trans_range=10):
        tr_x = trans_range * np.random.uniform() - trans_range/2
        tr_y = trans_range * np.random.uniform() - trans_range/2
        Trans_M = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
        self.warp_affine(Trans_M)
        
    def random_shear(self, shear_range=5):
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        Shear_M = cv2.getAffineTransform(pts1, pts2)
        self.warp_affine(Shear_M)
        
    def random_transform(self):
        intensity=self.flags['intensity']
        def _get_delta(intensity):
            delta = np.radians(intensity)
            rand_delta = np.random.uniform(low=-delta, high=delta)
            return rand_delta

        trans_M = AffineTransform(scale=(.9, .9),
                                 translation=(-_get_delta(intensity), _get_delta(intensity)),
                                 shear=_get_delta(intensity))
        self.img = warp(self.img, trans_M)
        self.label_img = warp(self.label_img, trans_M)
    
    def image_resize(self):
        self.img = cv2.resize(self.img, tuple(self.flags['size']), interpolation=cv2.INTER_CUBIC)
        self.label_img = cv2.resize(self.label_img, tuple(self.flags['size']), interpolation=cv2.INTER_CUBIC)
    
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
        self.img = plt.imread(os.path.join(self.file_path, self.raw_files[idx]))
        self.label_img = np.zeros_like(self.img)
        
        for lane_pt in lane_pts:
            cv2.polylines(self.label_img, np.int32([lane_pt]), isClosed=False, color=(255,255,255), thickness=15)   
            
        if self.flags['gray']:
            self.label_img = cv2.cvtColor(self.label_img, cv2.COLOR_BGR2GRAY)

    def __getitem__(self, idx):
        self.get_lane_image(idx)
        self.image_resize()
        self.preprocess()
#         self.random_brightness()
#         self.random_rotate()
#         self.random_translate()
#         self.random_shear()

        if self.flags['train']:
            self.random_transform()
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            self.label_img = np.array(self.label_img, dtype=np.uint8)
            return torch.FloatTensor(self.img), torch.LongTensor(self.label_img)
        else:
            self.img = img_as_float64(self.img)
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            return torch.FloatTensor(self.img)
    
    def __len__(self):
        return self.len
