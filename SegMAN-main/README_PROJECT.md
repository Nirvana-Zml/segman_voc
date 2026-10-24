# Pascal VOC2012 SegMAN 项目运行说明

本目录在官方 SegMAN/MMSegmentation 0.30.0 代码基础上增加了 VOC2012
训练、验证、TensorBoard、测试集推理和提交格式校验流程。

## 目录约定

容器 `segman_v2` 使用以下挂载：

```text
/workspace/segman                         SegMAN-main 源码
/workspace/segman/data/VOCdevkit/VOC2012 VOC2012 数据
/data/test400/test400/images              400 张测试图片
```

训练命令必须从 `/workspace/segman/segmentation` 启动。VOC 配置已将
`data_root` 设为 `../data/VOCdevkit/VOC2012`。

## 1. 数据审计

在仓库根目录执行：

```powershell
./SegMAN-main/scripts/audit_voc.ps1
```

只有审计退出码为 0 时才能开始正式训练。JSON 结果默认写入
`work_dirs/data_audit.json`。

## 2. 预训练权重

默认配置使用随机初始化，不会隐式下载权重。若课程允许 ImageNet-1K
分类预训练，将官方编码器权重放在例如：

```text
/workspace/segman/pretrained/SegMAN_Encoder_s.pth.tar
```

不要使用已经在 VOC、ADE20K 或 COCO 分割标注上训练的完整分割权重，
除非课程规则明确允许。记录权重来源和 SHA-256。

## 3. 训练

SegMAN-T 流程冒烟或基线：

```powershell
./SegMAN-main/scripts/train_voc_t.ps1
```

SegMAN-S 正式基线，使用允许的 ImageNet 编码器权重：

```powershell
./SegMAN-main/scripts/train_voc_s.ps1 `
  -Pretrained /workspace/segman/pretrained/SegMAN_Encoder_s.pth.tar
```

默认配置为 batch size 2、512×512 裁剪、55 个完整训练周期（40260
iterations）、AdamW、CE、
每 732 iterations 验证一次，并按验证集 mIoU 保存最佳权重。单卡训练时
MMSeg 会把 SyncBN 转为普通 BN；不要把训练 batch size 降为 1，因为解码器
的 1×1 全局池化分支无法用单样本更新 BN 统计。

启动长时间训练前可复现一次真实数据单步冒烟：

```powershell
./SegMAN-main/scripts/smoke_voc_s.ps1
```

随后运行两张真实 VOC 样本的确定性过拟合诊断：

```powershell
./SegMAN-main/scripts/overfit_voc_t.ps1
```

该诊断只证明图像、标签、损失和反向传播链路正确，不属于正式实验结果，
其 checkpoint 不得用于测试集提交。

## 4. TensorBoard

训练配置记录训练损失、学习率以及验证产生的 mIoU、mAcc 和 aAcc。当前
`segman_v2` 创建时没有映射 6006 端口；若需要从主机浏览器访问，重建或
重建后启动容器时增加 `-p 6006:6006`，然后执行：

```powershell
docker exec -w /workspace/segman segman_v2 `
  /root/anaconda3/bin/conda run -n segman `
  tensorboard --logdir work_dirs --bind_all --port 6006
```

也可以在主机安装 TensorBoard 后直接读取挂载目录：

```powershell
tensorboard --logdir SegMAN-main/work_dirs --port 6006
```

如需把固定验证样本在多个 checkpoint 上的预测变化写入 TensorBoard：

```powershell
docker exec -w /workspace/segman/segmentation segman_v2 `
  /root/anaconda3/bin/conda run --no-capture-output -n segman `
  python tools/log_val_predictions.py `
  local_configs/segman/voc/segman_s_voc12.py `
  /workspace/segman/work_dirs/segman_s_voc12 `
  --voc-root ../data/VOCdevkit/VOC2012 `
  --sample-ids 2007_000033 2007_000042 `
  --log-dir /workspace/segman/work_dirs/segman_s_voc12/val_images
```

固定样本 ID 必须来自 `ImageSets/Segmentation/val.txt`。TensorBoard 标量中的
loss 是训练损失；验证曲线使用 mIoU、mAcc 和 aAcc，报告中不要将其误标为
验证 loss。

## 5. 验证

```powershell
./SegMAN-main/scripts/eval_voc.ps1 `
  -Model s `
  -Checkpoint /workspace/segman/work_dirs/segman_s_voc12/best_mIoU_iter_XXXX.pth
```

## 6. 400 张测试图推理与格式校验

```powershell
./SegMAN-main/scripts/predict_test400.ps1 `
  -Model s `
  -Checkpoint /workspace/segman/work_dirs/segman_s_voc12/best_mIoU_iter_XXXX.pth
```

脚本随后自动检查：文件数、文件名、原图尺寸、PNG 模式、`uint8` 类型及
像素范围 `0..20`。预测目录只保存单通道类别索引图；彩色可视化不得混入。

## 7. 测试

```powershell
docker exec -w /workspace/segman segman_v2 `
  /root/anaconda3/bin/conda run --no-capture-output -n segman `
  python -m unittest discover -s tests -v
```

正式训练前还应完成少量样本过拟合测试。训练、验证和推理均通过后，再按
课程要求打包源码、报告和恰好 400 张预测 PNG。
