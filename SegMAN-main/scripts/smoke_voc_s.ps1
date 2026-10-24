param(
    [string]$Container = 'segman_v2',
    [string]$WorkDir = '/workspace/segman/work_dirs/smoke_s',
    [int]$Seed = 42
)

$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/train.py',
    'local_configs/segman/voc/segman_s_voc12.py',
    '--work-dir', $WorkDir,
    '--seed', $Seed,
    '--deterministic',
    '--cfg-options',
    'runner.max_iters=1',
    'evaluation.interval=100',
    'checkpoint_config.interval=1',
    'log_config.interval=1',
    'data.workers_per_gpu=1'
)
& docker @arguments
exit $LASTEXITCODE
