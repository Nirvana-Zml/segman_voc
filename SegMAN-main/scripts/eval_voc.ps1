param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet('t', 's')][string]$Model = 's',
    [string]$Container = 'segman_v2'
)

$config = "local_configs/segman/voc/segman_${Model}_voc12.py"
$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/test.py', $config,
    '--checkpoint', $Checkpoint,
    '--eval', 'mIoU'
)
& docker @arguments
exit $LASTEXITCODE
