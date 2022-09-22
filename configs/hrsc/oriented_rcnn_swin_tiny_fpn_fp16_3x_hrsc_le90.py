_base_ = ['./oriented_rcnn_swin_tiny_fpn_3x_hrsc_le90.py']

fp16 = dict(loss_scale='dynamic')