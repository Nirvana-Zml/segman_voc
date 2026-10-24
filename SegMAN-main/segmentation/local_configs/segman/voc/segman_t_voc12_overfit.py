_base_ = ['./segman_t_voc12.py']

# Diagnostic-only deterministic pipeline for proving that the data, labels,
# loss and optimizer can overfit two real VOC samples. Do not report this run
# as a validation result or use its checkpoint for submission.
smoke_split = (
    '/workspace/segman/segmentation/local_configs/segman/voc/smoke_train.txt')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    # Keep the transform to populate the `flip` metadata expected by Collect,
    # while making the diagnostic pipeline deterministic.
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(split=smoke_split, pipeline=train_pipeline))

optimizer = dict(lr=1e-4)
lr_config = dict(
    _delete_=True,
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=100)
evaluation = dict(interval=101, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=100, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
