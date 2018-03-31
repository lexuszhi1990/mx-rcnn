# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import csv
import os
import pickle
import numpy as np

from ..logger import logger
from .imdb import IMDB
from .ds_utils import unique_boxes

class FashionKeypoint(IMDB):
    def __init__(self, image_set, root_path, data_path, is_dev=False):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param data_path: data and results
        :return: imdb object
        """
        super(FashionKeypoint, self).__init__('fashion_keypoint', image_set, root_path, data_path)

        self.is_dev = is_dev

        self.image_set = image_set
        self.root_path = root_path
        self.data_path = data_path

        self.classes = ['__background__', 'blouse', 'skirt', 'outwear', 'dress', 'trousers']
        self.class_id = [0, 1, 2, 3, 4, 5]
        self.keypoints_name = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print ('num_images', self.num_images)

    def load_image_set_index(self):
        # field names :
        # ['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

        data_path = Path(self.data_path)
        train_anno = data_path.joinpath('Annotations/annotations.csv')
        train_anno_op = open(train_anno.as_posix(), 'r')
        csv_reader = csv.DictReader(train_anno_op)

        return [row['image_id'] for row in csv_reader]

    def image_path_from_index(self, index):
        """ example: DATA_PATH / Images/blouse/d21eab37ddc74ea5a5f1b4a5d3d9055a.jpg """
        image_path = Path(self.data_path, index)
        assert image_path.exists(), 'Path does not exist: {}'.format(image_path)
        return image_path.as_posix()

    def gt_roidb(self):
        # cache_file = Path(self.cache_path, self.name + '_gt_roidb.pkl')
        # if cache_file.exists():
        #     with open(cache_file.as_posix(), 'rb') as f:
        #         roidb = cPickle.load(f)
        #     print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb
        # gt_roidb = self.load_fashion_kp_annotations()
        # with open(cache_file, 'wb') as f:
        #     cPickle.dump(gt_roidb, f)

        train_anno = Path(self.data_path, 'Annotations/annotations.csv')
        train_anno_op = open(train_anno.as_posix(), 'r')
        csv_reader = csv.DictReader(train_anno_op)
        gt_roidb = [self.load_fashion_kp_annotations(row) for row in csv_reader]

        return gt_roidb

    def load_fashion_kp_annotations(self, anno_dict):

        num_objs = 1
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        image_path = self.image_path_from_index(anno_dict['image_id'])
        image_raw = cv2.imread(image_path)
        height, width, _ = image_raw.shape

        category = anno_dict['image_category']
        anno_kp = np.array([anno_dict.get(key).split('_') for key in self.keypoints_name]).astype(np.int16)
        anno_kp = anno_kp[np.where(anno_kp[:, 2]>=0)].astype(np.uint16)
        xmax, ymax, _ = anno_kp.max(axis=0)
        xmin, ymin, _ = anno_kp.min(axis=0)
        boxes[0, :] = [xmin, ymin, xmax, ymax]
        obj_cls = self.classes.index(category)
        gt_classes[0] = obj_cls
        overlaps[0, obj_cls] = 1.0

        roi_rec = {'image': image_path,
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        if self.is_dev:
            print(roi_rec)
        return roi_rec

