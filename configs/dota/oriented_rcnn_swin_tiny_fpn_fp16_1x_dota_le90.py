_base_ = ['./oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py']

fp16 = dict(loss_scale='dynamic')