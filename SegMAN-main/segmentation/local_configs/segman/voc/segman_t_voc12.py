_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/pascal_voc12.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_adamw.py',
]

# The container mounts VOCdevkit at /workspace/segman/data while training is
# launched from /workspace/segman/segmentation.
data_root = '../data/VOCdevkit/VOC2012'
samples_per_gpu = 2
iters_per_epoch = 732  # ceil(1464 / samples_per_gpu)

norm_cfg = dict(type='SyncBN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # SegMANDecoder uses pixel_unshuffle and therefore requires the
            # resized input dimensions to be divisible by 32.
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained=None,
        style='pytorch'),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=21,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=1,
    train=dict(data_root=data_root, reduce_zero_label=False),
    val=dict(
        data_root=data_root,
        reduce_zero_label=False,
        pipeline=test_pipeline),
    test=dict(
        data_root=data_root,
        reduce_zero_label=False,
        pipeline=test_pipeline),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
        'head': dict(lr_mult=10.0),
    }))
optimizer_config = dict()
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# Fifty-five complete passes over the 1464-image train split. Keeping the
# maximum aligned with iters_per_epoch guarantees a final validation run.
runner = dict(type='IterBasedRunner', max_iters=iters_per_epoch * 55)
evaluation = dict(
    interval=iters_per_epoch,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    rule='greater')
checkpoint_config = dict(
    by_epoch=False,
    interval=iters_per_epoch * 5,
    max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])

cudnn_benchmark = False
workflow = [('train', 1)]
