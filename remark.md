# 笔记

# 图像格式处理

原图队列类型 list[np.ndarray] np.uint8
原图的维度 (h, w, c)

处理后图队列类型 list[np.ndarray] np.float32
处理后图的维度 (h_new, w_new, c)

处理图像的逻辑：

```python
# 设置目标输入的短边长度
stdet_input_shortside = 256

# 图像的高宽
h, w, _ = image.shape

# 根据给定的短边长度stdet_input_shortside，重新计算图像的大小，保持宽高比不变
self.stdet_input_size = mmcv.rescale_size((w, h), (stdet_input_shortside, np.Inf))

# 设置图像归一化的参数，包括均值(mean)、标准差(std)，以及是否转换为RGB格式(to_rgb)
self.img_norm_cfg = dict(
    mean=np.array(config.model.data_preprocessor.mean),
    std=np.array(config.model.data_preprocessor.std),
    to_rgb=False)

# 将读取的帧调整到显示大小，并追加到images列表中
images.append(mmcv.imresize(image, self.display_size))

# 将帧调整到模型输入的大小，并转换为float32类型
processed_image = mmcv.imresize(image, self.stdet_input_size).astype(np.float32)

# 使用指定的均值和标准差对调整大小后的帧进行归一化处理
_ = mmcv.imnormalize_(processed_image, **self.img_norm_cfg)

# 将处理后的帧追加到processed_images列表中
processed_images.append(processed_image)
```

## mmdet 结果格式处理

`bboxs` 格式：np.ndarray 二维数组  float32
内部储存的形式
[
    [xyxy],
    [xyxy]
]
未归一化

处理逻辑：

```python
self.device = torch.device(device)

def predict(self, task):
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

# task.add_bboxes
def add_bboxes(self, display_bboxes):
    """Add correspondding bounding boxes."""
    self.display_bboxes = display_bboxes
    self.stdet_bboxes = display_bboxes.clone()
    self.stdet_bboxes[:, ::2] = self.stdet_bboxes[:, ::2] * self.ratio[0]
    self.stdet_bboxes[:, 1::2] = self.stdet_bboxes[:, 1::2] * self.ratio[1]

# 其中ratio计算方法
self.ratio = tuple(
            n / o for n, o in zip(self.stdet_input_size, self.display_size))
```