param(
    [string]$Container = 'segman_v2',
    [string]$Output = '/workspace/segman/work_dirs/data_audit.json'
)

$arguments = @(
    'exec', '-w', '/workspace/segman/segmentation', $Container,
    '/root/anaconda3/bin/conda', 'run', '--no-capture-output', '-n', 'segman',
    'python', 'tools/audit_voc.py',
    '../data/VOCdevkit/VOC2012',
    '--test-image-dir', '/data/test400/test400/images',
    '--output', $Output
)
& docker @arguments
exit $LASTEXITCODE
