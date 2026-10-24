param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet('t', 's')][string]$Model = 's',
    [string]$Container = 'segman_v2',
    [string]$OutputDir = '/workspace/segman/predictions/test400',
    [switch]$Overwrite
)

$config = "local_configs/segman/voc/segman_${Model}_voc12.py"
$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/predict_test400.py',
    $config, $Checkpoint,
    '/data/test400/test400/images', $OutputDir
)
if ($Overwrite) {
    $arguments += '--overwrite'
}
& docker @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$verifyArguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/verify_predictions.py',
    '/data/test400/test400/images', $OutputDir,
    '--expected-count', '400'
)
& docker @verifyArguments
exit $LASTEXITCODE
