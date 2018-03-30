
### install deps for image

```
apt-get install -y python3.6-dev
pip3.6 install opencv-python easydict cython
```

### start docker with mask rcnnn image

docker run --rm -it mxnet-cu90/python:1.2.0-roialign

docker run --rm -it -v /home/david/fashionAI:/app-dev -v /data/fashion/data/keypoint/warm_up_train_20180222:/warm_up_trin mxnet-cu90/python:1.2.0-roialign

docker run --rm -it -v /home/david/fashionAI/rcnn:/app-dev -v /data/fashion/data/keypoint/warm_up_train_20180222:/mnt/warm_up_trin -v /data/david/cocoapi:/mnt/coco mxnet-cu90/python:1.2.0-roialign


