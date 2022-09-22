_base_ = ['./oriented_rcnn_r50_pafpn_1x_dotav1.0_ms_512_le90_mae.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth' #'/home/server/WPW/Remote/Projects/mmrotate/checkpoints/vit-b-checkpoint-1599.pth'  # '/home/server/WPW/Remote/Projects/mmrotate/checkpoints/vitae-b-checkpoint-1599-transform-no-average.pth'

# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims= 96, #768, #96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7, #12,  # tiny 7     samall 7
        mlp_ratio=4, #4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3), #(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)

        # hukaixuan config
        # type='SwinTransformer',
        # embed_dims=96,  # tiny 96    small 96        base 128       large 192
        # depths=[2, 2, 6, 2],  # tiny 2262  small 2 2 18 2  base 2 2 18 2  large 2 2 18 2
        # num_heads=[3, 6, 12, 24],
        # window_size=7,  # tiny 7     samall 7
        # mlp_ratio=4.,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.2,  # 训练时间小于1×最好置为0.1
        # # ape=False,  # 是否需要对嵌入向量进行相对位置编码
        # patch_norm=True,
        # out_indices=(1, 2, 3),  # strides: [4, 8, 16, 32]  channel:[96, 192, 384, 768]
        # # use_checkpoint=False,  # 为True则显存消耗量更少，但前向传播次数加倍/ 为False则正常训练

        # type='SwinTransformer',
        # embed_dims=96,
        # depths=[2, 2, 6, 2],
        # num_heads=[3, 6, 12, 24],
        # window_size=7,
        # mlp_ratio=4,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.2,
        # patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        # with_cp=False,
        # convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        _delete_=True,
        type='PAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg

        # hukaixaun config
        # type='PAFPN',
        # in_channels=[384, 768, 1536],
        # out_channels=256,
        # # start_level=1,
        # add_extra_convs='on_input',
        # num_outs=5,
        # norm_cfg=norm_cfg

        # type='PAFPN',
        # in_channels=[192, 384, 768, 1536],
        # out_channels=256,
        # # start_level=1,
        # add_extra_convs='on_input',
        # num_outs=5,
        # norm_cfg=norm_cfg
    )


)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
