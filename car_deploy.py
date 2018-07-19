import argparse
import csv
from pathlib import Path
import ast
import pprint
import time

import mxnet as mx
from mxnet.module import Module

from data.bbox import im_detect
from data.loader import load_test, generate_batch
from data.vis import vis_detection
from net.model import load_param, check_shape
from utils.args import parse_args


def demo_net(sym, class_names, args, result_path):
    # print config
    print('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup context
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # load single test
    im_tensor, im_info, im_orig = load_test(args.image, short=args.img_short_side, max_size=args.img_long_side,
                                            mean=args.img_pixel_means, std=args.img_pixel_stds)

    # generate data batch
    data_batch = generate_batch(im_tensor, im_info)

    # load params
    arg_params, aux_params = load_param(args.params, ctx=ctx)

    # produce shape max possible
    data_names = ['data', 'im_info']
    label_names = None
    data_shapes = [('data', (1, 3, args.img_long_side, args.img_long_side)), ('im_info', (1, 3))]
    label_shapes = None

    # check shapes
    check_shape(sym, data_shapes, arg_params, aux_params)

    # create and bind module
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # forward
    forward_starts = time.time()
    mod.forward(data_batch)
    rois, scores, bbox_deltas = mod.get_outputs()
    rois.wait_to_read()
    rois = rois[:, 1:]
    scores = scores[0]
    bbox_deltas = bbox_deltas[0]
    forward_costs = time.time() - forward_starts
    print("forward costs %.4f" % (forward_costs))

    im_info = im_info[0]
    # decode detection
    det = im_detect(rois, scores, bbox_deltas, im_info,
                    bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh,
                    conf_thresh=args.rcnn_conf_thresh)


    fieldnames = ['name', 'coordinate']
    if result_path.exists():
        csvfile = result_path.open("a")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    else:
        csvfile = result_path.open("w+")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    img_name = Path(args.image).name
    bbox_str = ''
    for [cls, conf, x1, y1, x2, y2] in det:
        if cls > 0 and conf > args.vis_thresh:
            print(class_names[int(cls)], conf, [x1, y1, x2, y2])
            bbox_str += "%d_%d_%d_%d;" % (int(x1), int(y1), int(x2-x1), int(y2-y1))
    writer.writerow({'name': img_name, 'coordinate': bbox_str[:-1]})
    csvfile.close()
    print("detect image %s" % img_name) 

    # if vis
    if args.vis:
        vis_detection(im_orig, det, class_names, thresh=args.vis_thresh, prefix=args.image)


def get_resnet101_test(args):
    from net.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'ckpt/resnet101-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))




def main():
    args = parse_args()
    sym = get_resnet101_test(args)


    data_path = "/mnt/dataset/car"
    label_path = Path(data_path, "annotations", "%s.csv" % args.imageset)
    assert label_path.exists(), "label_path not exists %s " % label_path

    result_path = Path('%s-%s.csv' % ("results", "%s"%(time.strftime("%Y-%m-%d-%H-%M"))))
    print("create results file: %s" % result_path) 
    
    with label_path.open('r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = Path(data_path, "images", args.imageset, row['name'])
            assert img_path.exists(), "img_path not exists %s " % img_path
            args.image = img_path.as_posix()
            demo_net(sym, ["__BG__", "car"], args, result_path)


if __name__ == '__main__':
    main()
