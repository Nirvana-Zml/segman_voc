from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('mmcv'), 'mmcv is not installed')
class VocConfigTest(unittest.TestCase):
    def test_voc_configs_have_required_contract(self) -> None:
        from mmcv import Config

        config_root = (
            Path(__file__).resolve().parents[1]
            / 'segmentation' / 'local_configs' / 'segman' / 'voc'
        )
        for name in ('segman_t_voc12.py', 'segman_s_voc12.py'):
            with self.subTest(config=name):
                config = Config.fromfile(config_root / name)
                self.assertEqual(config.model.decode_head.num_classes, 21)
                self.assertEqual(config.model.decode_head.ignore_index, 255)
                self.assertTrue(
                    config.model.decode_head.loss_decode.avg_non_ignore)
                self.assertEqual(config.data.samples_per_gpu, 2)
                self.assertFalse(config.data.train.reduce_zero_label)
                self.assertEqual(config.evaluation.interval, 732)
                self.assertEqual(config.evaluation.save_best, 'mIoU')
                transforms = config.data.test.pipeline[1].transforms
                pads = [item for item in transforms if item.type == 'Pad']
                self.assertEqual(len(pads), 1)
                self.assertEqual(pads[0].size_divisor, 32)
                hook_types = {hook.type for hook in config.log_config.hooks}
                self.assertIn('TensorboardLoggerHook', hook_types)

        overfit = Config.fromfile(config_root / 'segman_t_voc12_overfit.py')
        self.assertEqual(overfit.runner.max_iters, 100)
        self.assertEqual(overfit.data.samples_per_gpu, 2)
        self.assertTrue(Path(overfit.data.train.split).is_absolute())


if __name__ == '__main__':
    unittest.main()
