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

docker run --rm -it -v /home/david/fashionAI/mx-rcnn:/app -v /data/david/cocoapi:/mnt/data/coco mxnet/python:1.2.0_gpu_cuda9-dev

train:
```
python3 train.py --pretrained ckpt/resnet_coco-0010.params --network resnet101 --dataset coco --imageset person_train2017 --rcnn_num_classes 2 --gpus 1
```

validate:
```
python3 test.py --network resnet101 --dataset coco --imageset person_val2017 --params model/resnet101-0002.params --gpu 4

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.627
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.097
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.657
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.167
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.403
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
```

demo:
```
python3 demo.py --network resnet101 --dataset coco --imageset person_val2017 --params model/resnet101-0002.params --gpu 5 --vis --image samples/demo/1045023827_4ec3e8ba5c_z.jpg
```

deploy:
```
python3 deploy.py --network resnet101 --rcnn_num_classes 2 --prefix ./model/resnet101 --epoch 3 --rpn-post-nms-topk 100
```

