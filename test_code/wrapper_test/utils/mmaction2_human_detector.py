from abc import ABCMeta, abstractmethod

import numpy as np
import torch

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

from test_code.wrapper_test.utils import Mmaction2TaskInfo

class BaseHumanDetector(metaclass=ABCMeta):
    """Base class for Human Dector.

    Args:
        device (str): CPU/CUDA device option.
    """

    def __init__(self, device):
        self.device = torch.device(device)

    @abstractmethod
    def _do_detect(self, image):
        """Get human bboxes with shape [n, 4].

        The format of bboxes is (xmin, ymin, xmax, ymax) in pixels.
        """

    def predict(self, task: Mmaction2TaskInfo):
        """Add keyframe bboxes to task."""
        # keyframe idx == (clip_len * frame_interval) // 2
        keyframe = task.frames[len(task.frames) // 2]

        # call detector
        bboxes = self._do_detect(keyframe)

        # convert bboxes to torch.Tensor and move to target device
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(self.device)
        elif isinstance(bboxes, torch.Tensor) and bboxes.device != self.device:
            bboxes = bboxes.to(self.device)

        # update task
        task.add_bboxes(bboxes)

        return task


class Mmaction2HumanDetector(BaseHumanDetector):
    """Wrapper for mmdetection human detector.

    Args:
        config (str): Path to mmdetection config.
        ckpt (str): Path to mmdetection checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human detection score.
        person_classid (int): Choose class from detection results.
            Default: 0. Suitable for COCO pretrained models.
    """

    def __init__(self, config, ckpt, device, score_thr, person_classid=0):
        super().__init__(device)
        self.model = init_detector(config, ckpt, device=device)
        self.person_classid = person_classid
        self.score_thr = score_thr

    def _do_detect(self, image):
        """Get bboxes in shape [n, 4] and values in pixels."""
        det_data_sample = inference_detector(self.model, image)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        # We only keep human detection bboxs with score larger
        # than `det_score_thr` and category id equal to `det_cat_id`.
        valid_idx = np.logical_and(pred_instance.labels == self.person_classid,
                                   pred_instance.scores > self.score_thr)
        bboxes = pred_instance.bboxes[valid_idx]
        # result = result[result[:, 4] >= self.score_thr][:, :4]
        return bboxes