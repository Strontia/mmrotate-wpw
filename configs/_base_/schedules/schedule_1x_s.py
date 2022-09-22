# evaluation
evaluation = dict(# 训练期间做验证的设置
    interval=1,  # 执行验证的间隔
    metric='mAP' # 验证方法
)
# optimizer # 优化器设置
optimizer = dict(
    # 构建优化器的设置，支持：
    # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
    # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
    # 更多细节可参考 "tutorials/5_new_modules.md" 部分
    type='SGD',# 优化器类型, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13
    lr=0.0025,  # 学习率, 参数的细节使用可参考 PyTorch 的对应文档
    momentum=0.9,  # 动量大小
    weight_decay=0.0001  # SGD 优化器权重衰减
)
optimizer_config = dict(  # 用于构建优化器钩子的设置
    grad_clip=dict(max_norm=35, norm_type=2)  # 使用梯度裁剪
)
# learning policy
# 学习策略设置
lr_config = dict(  # 用于注册学习率调整钩子的设置
    policy='step',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11]  # 学习率衰减步长
)
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(  # 模型权重钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=1  # 模型权重文件保存间隔
 )
