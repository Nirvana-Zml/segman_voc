param(
    [string]$Container = 'segman_v2',
    [string]$WorkDir = '/workspace/segman/work_dirs/segman_s_voc12',
    [string]$Pretrained = '',
    [int]$Seed = 42
)

$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/train.py',
    'local_configs/segman/voc/segman_s_voc12.py',
    '--work-dir', $WorkDir,
    '--seed', $Seed,
    '--deterministic'
)
if ($Pretrained) {
    $arguments += '--cfg-options'
    $arguments += "model.backbone.pretrained=$Pretrained"
}
& docker @arguments
exit $LASTEXITCODE
