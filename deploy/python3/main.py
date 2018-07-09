# -*- coding: utf-8 -*-

import sys
import cv2
import mxnet as mx
import numpy as np
import random
from pathlib import Path
import time

from utils import im_detect


millisecond = lambda x: int(round(x * 1000))

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    img_path : str
        image path
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    threshold: float
        thresh for scores
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, model_prefix, epoch, shape=300, mean_pixels=(123, 117, 104), threshold=0.2, batch_size=1, ctx=None):
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.shape = shape
        self.threshold = threshold
        self.mean_pixels = mean_pixels
        self.batch_size = batch_size
        self.load_symbol, self.args, self.auxs = mx.model.load_checkpoint(model_prefix, epoch)
        self.args, self.auxs = self.ch_dev(self.args, self.auxs, self.ctx)

    def ch_dev(self, arg_params, aux_params, ctx):
        new_args = dict()
        new_auxs = dict()
        for k, v in arg_params.items():
            new_args[k] = v.as_in_context(ctx)
        for k, v in aux_params.items():
            new_auxs[k] = v.as_in_context(ctx)
        return new_args, new_auxs

    def make_input(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        short = 600
        max_size = 1000
        im_scale = float(short) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        mean = (0., 0., 0.)
        std = (1., 1., 1.)
        im_tensor = np.zeros((3, img.shape[0], img.shape[1]))
        for i in range(3):
            im_tensor[i, :, :] = (img[:, :, 2 - i] - mean[i]) / std[i]


        height, width = im_shape[:2]
        im_info = mx.nd.array([height, width, im_scale]).expand_dims(0)

        return img, im_info

    def detect(self, img_path):
        """
        wrapper for detecting multiple images

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        start = time.clock()
        im_data, im_info = self.make_input(img_path)
        print("make inputs costs: %dms" % millisecond(time.clock()-start))

        start = time.clock()
        self.args["data"] = mx.nd.array(im_data, self.ctx)
        self.args["im_info"] = mx.nd.array(im_info, self.ctx)
        exe = self.load_symbol.bind(self.ctx, self.args, args_grad=None, grad_req="null", aux_states=self.auxs)
        print("bind data  costs: %dms" % millisecond(time.clock()-start))

        start = time.clock()
        exe.forward()
        rois, scores, bbox_deltas = exe.outputs
        # https://github.com/apache/incubator-mxnet/issues/6974
        rois.wait_to_read()
        print("forward costs: %dms" % millisecond(time.clock()-start))

        rois = rois[:, 1:]
        scores = scores[0]
        bbox_deltas = bbox_deltas[0]

        im_info = im_info[0]
        # decode detection
        self.dets = im_detect(rois, scores, bbox_deltas, im_info, bbox_stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.3, conf_thresh=1e-3)

        for [cls, conf, x1, y1, x2, y2] in self.dets:
            if cls > 0 and conf > 0.5:
                print(int(cls), conf, [x1, y1, x2, y2])

    def save_results(self, img_path, save_path="./", color='red'):
        draw = cv2.imread(img_path)
        colors = dict()

        for det in self.dets:
            box = det[2:]
            score = det[1]
            cls_id = det[0]

            if score < 0.75:
                continue

            if cls_id not in colors:
                colors[cls_id] = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            left, top = int(box[0]), int(box[1])
            right, bottom = int(box[2]), int(box[3])
            cv2.rectangle(draw, (left, top), (right, bottom), colors[cls_id], 1)
            cv2.putText(draw, '%.3f'%score, (left, top+30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 1)

        img_path = Path(img_path)
        save_path = Path(img_path).parent.joinpath(img_path.stem + '_rcnn_deploy.png')
        cv2.imwrite(save_path.as_posix(), draw)
        print("save results at %s" % save_path)

def main(img_path):
    if not Path(img_path).exists():
        print(img_path+' image not exists')
        return

    ctx = mx.gpu(7)
    model_prefix = '/app/model/resnet101_deploy'
    epoch = 0
    threshold = 0.65
    shape = 300

    start = time.clock()
    ped_detector = Detector(model_prefix=model_prefix, epoch=epoch, threshold=threshold, shape=shape, ctx=ctx)
    ped_detector.detect(img_path)
    print("total time used: %.4fs" % (time.clock()-start))
    ped_detector.save_results(img_path)

if __name__ == '__main__':
    main(sys.argv[1])

