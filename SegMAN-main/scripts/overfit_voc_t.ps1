param(
    [string]$Container = 'segman_v2',
    [string]$WorkDir = '/workspace/segman/work_dirs/overfit_t',
    [int]$Seed = 42
)

$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/train.py',
    'local_configs/segman/voc/segman_t_voc12_overfit.py',
    '--work-dir', $WorkDir,
    '--seed', $Seed,
    '--deterministic'
)
& docker @arguments
exit $LASTEXITCODE
