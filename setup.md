### install deps for image

```
apt-get install -y python3.6-dev python3-tk
pip3.6 install opencv-python easydict cython matplotlib scikit-image
```

ipython config:
```
%load_ext autoreload
%autoreload 2
```

### start docker with mask rcnnn image

docker run --rm -it mxnet-cu90/python:1.2.0-roialign

on 172:
```
# for kp
docker run --rm -it -v /home/david/fashionAI/mx-rcnn:/app -v /data/david/fai_kp:/mnt/data/fai_kp -v /data/david/models/fai:/mnt/models -v /data/david/cocoapi:/mnt/data/coco mxnet-cu90/python:1.2.0-roialign

# for attr:
docker run --rm -it -v /home/david/fashionAI/mx-rcnn:/app-dev -v /data/david/fai_attr:/mnt/data/fai_attr -v /data/david/models/fai:/mnt/models -v /data/david/cocoapi:/mnt/data/coco mxnet-cu90/python:1.2.0-roialign
```

on 177:
```
docker run --rm -it -v /home/fulingzhi/workspace/mx-rcnn:/app-dev -v /mnt/gf_mnt/models:/mnt/models -v /mnt/gf_mnt/datasets/cocoapi:/mnt/data/coco -v /mnt/gf_mnt/datasets/VOCdevkit:/mnt/data/VOCdevkit mxnet-cu90/python:1.2.0-roialign
```

### local dev

docker run --rm -it -v /Users/david/repo/detection/mx-rcnn:/app-dev -v /Users/david/mnt/data/VOCdevkit:/mnt/data/VOCdevkit -v /Users/david/mnt/data/ckpt:/mnt/ckpt mxnet/python:1.2.0-dev bash


### logs

INFO:root:num_images 12534
INFO:root:save the gt roidb at /mnt/warm_up_train/build/train_gt_roidb.pkl
INFO:root:load data: filtered 0 roidb entries: 12534 -> 12534
INFO:root:providing maximum shape [('data', (1, 3, 312, 512)), ('gt_boxes', (1, 100, 5))] [('label', (1, 5472)), ('bbox_target', (1, 36, 19, 32)), ('bbox_weight', (1, 36, 19, 32))]
INFO:root:output shape {'bbox_loss_reshape_output': (1, 128, 24),
 'blockgrad0_output': (1, 128),
 'cls_prob_reshape_output': (1, 128, 6),
 'rpn_bbox_loss_output': (1, 36, 19, 19),
 'rpn_cls_prob_output': (1, 2, 171, 19)}
INFO:root:lr 0.001000 lr_epoch_diff [7] lr_iters [87738]


### demo

python3.6 demo.py --prefix model/final --epoch 0 --image pant_length_1.jpg --gpu 0
