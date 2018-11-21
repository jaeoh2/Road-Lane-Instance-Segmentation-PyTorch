#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import argparse

import torch
from enet import ENet

from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN


def coloring(mask, gray=False):
    # refer from : https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/utils.py
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        if gray:
            ins_color_img[mask == i + 1] = (np.array([255,255,255]).astype(np.uint8)
        else:
            ins_color_img[mask == i + 1] = (np.array(colors[i][:3]) * 255).astype(np.uint8)
            
    return ins_color_img


def gen_instance_mask(sem_pred, ins_pred, n_obj):
    embeddings = ins_pred[:, sem_pred].transpose(1, 0)
    clustering = DBSCAN(eps=0.05).fit(embeddings)
    labels = clustering.labels_

    instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
    for i in range(n_obj):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[sem_pred] += lbl

    return instance_mask


class LaneDetectNode(object):
    def __init__(self, model, args=None):
        rospy.init_node('road_lane_detection', anonymous=True)
        self.cvbridge = CvBridge()
        self.model = model
        self.img = None
        self.sem = None
        self.ins = None
        self.out = None
        self.args = args
        self.image_lock = threading.RLock()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw/',
                                          Image,
                                          callback=self.sub_lane,
                                          queue_size=1,
                                          buff_size=2**16)
        self.pub = rospy.Publisher('/lane_image', Image, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.get_instances)
        rospy.loginfo("road rain detection started")

    def preprocess(self, img):
        self.img = self.cvbridge.imgmsg_to_cv2(img, 'rgb8')
        self.img = cv2.resize(self.img, (224,224), interpolation=cv2.INTER_CUBIC)

        # CLAHE nomalization
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        img_plane = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_plane[0] = clahe.apply(img_plane[0])
        img = cv2.merge(img_plane)
        self.img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        self.img = np.array(np.transpose(self.img, (2, 0, 1)), dtype=np.float32)

    def sub_lane(self, img):
        if self.image_lock.acquire(True):

            self.preprocess(img)

            img_tensor = torch.from_numpy(self.img).unsqueeze(dim=0).cuda()
            sem_pred, ins_pred = self.model(img_tensor)
            sem_pred = sem_pred[:,1,:,:].squeeze(dim=0).cpu().data.numpy()
            sem_pred = ndi.morphology.binary_fill_holes(sem_pred > 0.5)
            ins_pred = ins_pred.squeeze(dim=0).cpu().data.numpy()

            self.out = coloring(gen_instance_mask(sem_pred, ins_pred, 8))
            self.out = self.cvbridge.cv2_to_imgmsg(self.out, encoding='rgb8')

            self.image_lock.release()

    def get_instances(self, event):
        if self.out is None:
            return

        self.out.header.stamp = rospy.Time.now()
        self.pub.publish(self.out)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path', required=True)
        args = parser.parse_args()

        model = ENet(input_ch=3, output_ch=2).cuda()
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        node = LaneDetectNode(model, args)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
