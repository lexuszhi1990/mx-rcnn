import sys
sys.path.append('/app-dev')

from rcnn.dataset import FashionKeypoint
fk = FashionKeypoint('None', root_path='/mnt/warm_up_trin/build', data_path='/mnt/warm_up_trin', is_dev=True)
fk.gt_roidb()


# legancy
# if category == 'blouse':
#     anno_keys = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']
# elif category == 'skirt':
#     anno_keys = ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right']
# elif category == 'outwear':
#     anno_keys = ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']
# elif category == 'dress':
#     import pdb
#     pdb.set_trace()
# else:
#     import pdb
#     pdb.set_trace()
