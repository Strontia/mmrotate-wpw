_base_ = ['./oriented_rcnn_swin_tiny_fpn_1x_dotav1.0_512_le90.py']

fp16 = dict(loss_scale='dynamic')