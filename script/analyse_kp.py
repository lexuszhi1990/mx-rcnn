from collections import Counter
import sys
sys.path.append('./rcnn/pycocotools')
from pycocotools.coco import COCO

train_labels = '/mnt/data/fai_kp/ROUND2/train_0427/Annotations/train_0427.json'
coco=COCO(train_labels)
img_idx = coco.getImgIds()

img_id = img_idx[0]
annIds = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(annIds)

not_exists_kp = 0
hided_kp = 0
total_kp = 0

cat_id = 5
annIds = coco.getAnnIds(catIds=[cat_id], iscrowd=None)
annos = coco.loadAnns(annIds)

for anno in annos:
    kp_counts = Counter(anno['keypoints'][2::3])
    not_exists_kp += kp_counts[0]
    hided_kp += kp_counts[1]
    total_kp += len(anno['keypoints'])

print("%s, not_exists_kp/total_kp %.4f" % (coco.cats[cat_id]['name'], not_exists_kp/total_kp))
print("%s, hided_kp/total_kp %.4f" % (coco.cats[cat_id]['name'], hided_kp/total_kp))
