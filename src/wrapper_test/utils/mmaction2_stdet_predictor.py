SUBMODULE_DIR = '/workspace/mmaction2'

import torch
from mmdet.apis import init_detector
from src.utils import class_temporary_change_dir

@class_temporary_change_dir(SUBMODULE_DIR)
class Mmaction2StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        self.score_thr = score_thr

        # load model
        # config.model.backbone.pretrained = None
        # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        # load_checkpoint(model, checkpoint, map_location='cpu')
        # model.to(device)
        # model.eval()
        model = init_detector(config, checkpoint, device=device)
        self.model = model
        self.device = device

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}
        try:
            if config['data']['train']['custom_classes'] is not None:
                self.label_map = {
                    id + 1: self.label_map[cls]
                    for id, cls in enumerate(config['data']['train']
                                             ['custom_classes'])
                }
        except KeyError:
            pass

    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.stdet_bboxes) == 0:
            return task

        with torch.no_grad():
            result = self.model(**task.get_model_inputs(self.device))
        scores = result[0].pred_instances.scores
        # pack results of human detector and stdet
        preds = []
        for _ in range(task.stdet_bboxes.shape[0]):
            preds.append([])
        for class_id in range(scores.shape[1]):
            if class_id not in self.label_map:
                continue
            for bbox_id in range(task.stdet_bboxes.shape[0]):
                if scores[bbox_id][class_id] > self.score_thr:
                    preds[bbox_id].append((self.label_map[class_id],
                                           scores[bbox_id][class_id].item()))

        # update task
        # `preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        task.add_action_preds(preds)

        return task