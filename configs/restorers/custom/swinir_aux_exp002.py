exp_name = 'swinir_psnr_x4_g1_1000k_div2k_aux'

scale = 4
# model settings
model = dict(
    type='BasicAuxRestorer',
    generator=dict(
        type='SwinAuxIR',
        in_channels=3,
        img_size=48,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dims=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRAnnotationAuxDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='aux',
        flag='unchanged'),    
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'aux']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    #dict(type='PairedRandomCrop', gt_patch_size=96, keys=['lq', 'gt', 'aux']),
    dict(
        type='Flip', keys=['lq', 'gt', 'aux'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt', 'aux'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt', 'aux'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt', 'aux'], meta_keys=['lq_path', 'gt_path', 'aux_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'aux'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=32,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/DIV2K/DIV2K_train_LR_bicubic/X4_sub_jpg',
            gt_folder='data/DIV2K/DIV2K_train_HR_sub',
            ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='data/DIV2K/DIV2K_valid_LR_bicubic/X4',
        gt_folder='data/DIV2K/DIV2K_valid_HR',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='data/DIV2K/DIV2K_valid_LR_bicubic/X4',
        gt_folder='data/DIV2K/DIV2K_valid_HR',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
# keep training
total_iters = 100000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[25000, 40000, 45000, 475000, 50000],
    gamma=0.5)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = 'models/swinIR-M_x4.pth'
resume_from = None
workflow = [('train', 1)]

# # fp16 settings
# fp16 = dict(loss_scale=512.0)