SUBMODULE_DIR = '/workspace/mmaction2'

from mmaction2.mmaction.structures import ActionDataSample
from src.wrapper.utils import *
from src.utils import AutoFIFO, CircularQueue
from mmengine import Config
from mmengine.structures import InstanceData
import torch
from mmdet.apis import init_detector
from src.utils import temporary_change_dir
import mmcv
import numpy as np
import threading
import warnings
import traceback
import copy
from typing import Dict, List, Tuple
warnings.filterwarnings('always')

# step说明
# 0 完成行为识别
# 1 开始行为识别
# 2 完成添加目标检测的结果
# 3 开始添加目标检测的结果
# 4 完成添加图片
# 5 开始添加图片
# 6 完成添加image id
BEHIAVIOR_RECOGNIZE_COMPLETE = 0
BEHIAVIOR_RECOGNIZE_START = 1
ADD_BBOXES_COMPLETE = 2
ADD_BBOXES_START = 3
ADD_IMAGE_COMPLETE = 4
ADD_IMAGE_START = 5
ADD_IMAGE_ID_COMPLETE = 6

# @class_temporary_change_dir(SUBMODULE_DIR)
class Mmaction2Recognizer:
    class Mmaction2Builder:
        def __init__(self):
            """
            Initialize the class with default values for the arguments.
            """
            # spatio temporal detection config file path
            self.config: str = ''
            
            # spatio temporal detection checkpoint file/url
            self.checkpoint: str = ''
            
            # the threshold of human detection score
            self.det_score_thr: float = 0.9
            
            # the threshold of human action score
            self.action_score_thr: float = 0.4
            
            # label map file
            self.label_map: str = ''
            
            # CPU/CUDA device option
            self.device: str = 'cuda:0'
            
            # 设置目标输入的短边长度
            self.stdet_input_shortside = 256
            
        def build(self):
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    warnings.warn("cuda is not available", UserWarning)
                self.device = 'cpu'
            return Mmaction2Recognizer(self)
    
    class ImageInfo:
        image_id = 0
        image = None
        stdet_input_size: Tuple[int, int] = (0, 0)
        origin_person_bboxes: Dict[int, List[float]] = {}
        processed_person_bboxes: Dict[int, torch.Tensor] = {}
        preds: Dict[int, List[Tuple[str, float]]] = {}
        step = ADD_IMAGE_ID_COMPLETE
        lock = threading.Lock()
        
    def __init__(self, builder: Mmaction2Builder):
        # 加载配置
        self.__config: Config = self.__load_config(builder.config)
        
        # 检测设备
        self.__device_str: str = builder.device
        self.__device_torch: torch.device = torch.device(self.__device_str)
        
        # 行为识别
        self.__action_score_thr: float = builder.action_score_thr
        self.__checkpoint: str = builder.checkpoint
        self.__label_map: str = builder.label_map
        
        # 窗口大小
        val_pipeline = self.__config.val_pipeline
        sampler = [x for x in val_pipeline
                   if x['type'] == 'SampleAVAFrames'][0]
        clip_len = sampler['clip_len']
        self.__frame_interval = sampler['frame_interval']
        self.__window_size = clip_len * self.__frame_interval
        
        # 将所有的处理完毕的图片按照frame_interval为间隔，排成frame_interval行，每行clip_len个
        self.__processed_images: List[AutoFIFO] = [AutoFIFO(max_size=clip_len) for _ in range(self.__frame_interval)]
        
        # 当前图片要放进self.__processed_images第几行 先-1预处理，使用时先+1计算
        self.__frame_pos = self.__window_size // 2 - (clip_len // 2) * self.__frame_interval - 1
        
        # 根据窗口大小设置图像信息队列
        self.__image_infos: Dict[int, Mmaction2Recognizer.ImageInfo] = dict()
        self.__image_id_queue: CircularQueue = CircularQueue(self.__window_size)
        
        # 图像预处理时resize的短边长度
        self.__stdet_input_shortside: int = builder.stdet_input_shortside
        
        # 行为识别为有状态算法，每个任务独占一个识别器
        # 延迟加载
        self.__model: torch.nn.Module = None
        
        # 最后一个识别完成的图像
        self.latest_predict_completed_image_id: int = 0
        # 最新添加图片的uid
        self.latest_add_image_id: int = 0
        
    def __load_config(self, config_path: str) -> Config:
        """加载配置

        Args:
            config_path (str): 配置文件路径

        Returns:
            Config: 配置对象
        """
        # init action detector
        config = Config.fromfile(config_path)

        try:
            # In our spatiotemporal detection demo, different actions should have
            # the same number of bboxes.
            config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
        except KeyError:
            pass
        return config
    
    def load_model(self) -> bool:
        """加载模型

        Returns:
            bool: 是否加载成功
        """
        try:
            with temporary_change_dir(SUBMODULE_DIR):
                self.__model = init_detector(self.__config, self.__checkpoint, device=self.__device_str)

            # init label map, aka class_id to class_name dict
            with open(self.__label_map) as f:
                lines = f.readlines()
            lines = [x.strip().split(': ') for x in lines]
            self.__label_map = {int(x[0]): x[1] for x in lines}
            try:
                if self.__config['data']['train']['custom_classes'] is not None:
                    self.__label_map = {
                        id + 1: self.__label_map[cls]
                        for id, cls in enumerate(self.__config['data']['train']
                                                ['custom_classes'])
                    }
            except KeyError:
                pass
        except:
                traceback.print_exc()
                return False
        return True
    
    def check_image_id_exist(self, image_id: int) -> bool:
        """检查图像id是否存在

        Args:
            image_id (int): 图像id

        Returns:
            bool: 图像id是否存在
        """
        return image_id in self.__image_infos
    
    def get_step_by_image_id(self, image_id: int) -> int:
        """获取当前阶段
        Args:
            image_id (int): 图像id

        Returns:
            int: 阶段ID, 特殊的： -1图像不存在 0检测完成
        """
        if not self.check_image_id_exist(image_id):
            warnings.warn("image_id 不存在", UserWarning)
            return -1
        return self.__image_infos[image_id].step
    
    def add_image_id(self, image_id: int) -> bool:
        """添加图像id

        Args:
            image_id (int): 图像id

        Returns:
            bool: 是否添加成功
        """
        if self.check_image_id_exist(image_id):
            warnings.warn("image_id 重复添加", UserWarning)
            return False
        # 清理溢出
        if (len(self.__image_infos) >= self.__window_size):
            image_id_rm = self.__image_id_queue.dequeue()
            with self.__image_infos[image_id_rm].lock:
                self.__image_infos.pop(image_id_rm)
        self.__image_id_queue.enqueue(image_id)
        self.__image_infos[image_id] = Mmaction2Recognizer.ImageInfo()
        self.__image_infos[image_id].step = ADD_IMAGE_ID_COMPLETE
        self.latest_add_image_id = image_id
        return True
    
    def add_image(self, image_id: int, image: np.ndarray) -> bool:
        """添加图像

        Args:
            image_id (int): 图像的id
            image (np.ndarray): np.ndarray np.uint8 格式的图像数据

        Returns:
            bool: 是否添加成功
        """
        if not self.check_image_id_exist(image_id):
            self.add_image_id(image_id)
        elif self.__image_infos[image_id].step != ADD_IMAGE_ID_COMPLETE:
            print(self.__image_infos[image_id].step)
            warnings.warn("步骤顺序错误", UserWarning)
            return False
        
        # 标记状态
        self.__image_infos[image_id].step = ADD_IMAGE_START
        
        # 图像的高宽
        h, w, _ = image.shape

        # 根据给定的短边长度stdet_input_shortside，重新计算图像的大小，保持宽高比不变
        self.__image_infos[image_id].stdet_input_size = mmcv.rescale_size((w, h), (self.__stdet_input_shortside, np.Inf))

        # 设置图像归一化的参数，包括均值(mean)、标准差(std)，以及是否转换为RGB格式(to_rgb)
        self.img_norm_cfg = dict(
            mean=np.array(self.__config.model.data_preprocessor.mean),
            std=np.array(self.__config.model.data_preprocessor.std),
            to_rgb=False)

        # 存储image
        self.__image_infos[image_id].image = image

        # 将帧调整到模型输入的大小，并转换为float32类型
        processed_image = mmcv.imresize(image, self.__image_infos[image_id].stdet_input_size).astype(np.float32)

        # 使用指定的均值和标准差对调整大小后的帧进行归一化处理
        _ = mmcv.imnormalize_(processed_image, **self.img_norm_cfg)
        
        # 更新本张图片位置
        self.__frame_pos += 1
        self.__frame_pos %= self.__frame_interval
        
        # 储存处理后图片
        self.__processed_images[self.__frame_pos].push(processed_image)
        
        # 标记状态
        self.__image_infos[image_id].step = ADD_IMAGE_COMPLETE
        return True
    
    def add_person_bboxes(self, image_id: int, person_bboxes: Dict[int, List[float]]) -> bool:
        """添加人的bboxes

        Args:
            image_id (int): 图像id
            person_bboxes (Dict[int, List[float]]): 人的bboxes float32 格式： person_id: [x1, y1, x2, y2]

        Returns:
            bool: 是否添加成功
        """
        # 先检查步骤是否正确
        if not self.check_image_id_exist(image_id):
            warnings.warn(f"image id {image_id} not found", UserWarning)
            return False
        if self.__image_infos[image_id].step != ADD_IMAGE_COMPLETE:
            print(self.__image_infos[image_id].step)
            warnings.warn("步骤顺序错误", UserWarning)
            return False
        self.__image_infos[image_id].step = ADD_BBOXES_START
        self.__image_infos[image_id].origin_person_bboxes = copy.deepcopy(person_bboxes)
        h, w, _ = self.__image_infos[image_id].image.shape
        ratio = tuple(
            n / o for n, o in zip(self.__image_infos[image_id].stdet_input_size, (w, h)))
        # convert bboxes to torch.Tensor
        for person_id in person_bboxes: 
            person_bbox = np.array(person_bboxes[person_id], dtype=np.float32)
            bboxes = torch.from_numpy(person_bbox).to(self.__device_torch)
            bboxes[::2] = bboxes[::2] * ratio[0] * w
            bboxes[1::2] = bboxes[1::2] * ratio[1] * h
            self.__image_infos[image_id].processed_person_bboxes[person_id] = bboxes
        
        self.__image_infos[image_id].step = ADD_BBOXES_COMPLETE
        return True
    
    def get_key_image_id(self) -> int:
        """获取关键帧的图像id

        Returns:
            int: 关键帧的图像id, 如果是0则是还未生成关键帧
        """
        if len(self.__image_infos) != self.__window_size:
            warnings.warn(f"图像缓存数量不足，需要 {self.__window_size}，当前 {len(self.__image_infos)}", UserWarning)
            return 0
        key_image_id = self.__image_id_queue[self.__window_size // 2]
        return key_image_id
    
    def predict_by_image_id(self, key_image_id: int) -> bool:
        """根据图像id进行预测, 目前仅支持输入当前队列的关键帧

        Args:
            key_image_id (int): 关键帧的图像id

        Returns:
            bool: 是否检测成功
        """
        if not self.check_image_id_exist(key_image_id):
            warnings.warn(f"image id {key_image_id} not found", UserWarning)
            return False
        if self.__image_infos[key_image_id].step != ADD_BBOXES_COMPLETE:
            print(self.__image_infos[key_image_id].step)
            warnings.warn("步骤顺序错误", UserWarning)
            return False
        self.__image_infos[key_image_id].step = BEHIAVIOR_RECOGNIZE_START
        
        if len(self.__image_infos) != self.__window_size:
            warnings.warn(f"图像缓存数量不足，需要 {self.__window_size}，当前 {len(self.__image_infos)}", UserWarning)
            return False
        
        if 0 == len(self.__image_infos[key_image_id].processed_person_bboxes):
            return True
        
        with torch.no_grad():
            cur_frames = self.__processed_images[self.__frame_pos].to_list()
            input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(self.__device_str)
            
            datasample = ActionDataSample()
            bboxes = []
            key_image_info = self.__image_infos[key_image_id]
            stdet_input_w, stdet_input_h = key_image_info.stdet_input_size
            for bbox in key_image_info.processed_person_bboxes.values():
                bboxes.append(bbox.unsqueeze(0))
            bboxes_tensor = torch.cat(bboxes, dim=0)
            datasample.proposals = InstanceData(bboxes=bboxes_tensor)
            datasample.set_metainfo(dict(img_shape=(stdet_input_h, stdet_input_w)))
            model_input = dict(
                inputs=input_tensor, 
                data_samples=[datasample], 
                mode='predict')
            result = self.__model(**model_input)
        scores = result[0].pred_instances.scores
        preds = []
        for _ in range(bboxes_tensor.shape[0]):
            preds.append([])
        for class_id in range(scores.shape[1]):
            if class_id not in self.__label_map:
                continue
            for bbox_id in range(bboxes_tensor.shape[0]):
                if scores[bbox_id][class_id] > self.__action_score_thr:
                    preds[bbox_id].append((self.__label_map[class_id],
                                           scores[bbox_id][class_id].item()))
        for idx, person_id in enumerate(key_image_info.processed_person_bboxes.keys()):
            key_image_info.preds[person_id] = preds[idx]
        self.latest_predict_completed_image_id = key_image_id
        key_image_info.step = BEHIAVIOR_RECOGNIZE_COMPLETE
        return True
        
    def get_result_by_image_id(self, image_id: int) -> Dict[int, List[Tuple[str, float]]]:
        """根据图像id获取结果

        Args:
            image_id (int): 图像id

        Returns:
            Dict[int, List[Tuple[str, float]]]: 结果, 字典的key为person_id, 结果是Tuple(行为, 置信度)的列表
        """
        if not self.check_image_id_exist(image_id):
            warnings.warn(f"image id {image_id} not found", UserWarning)
            return {}
        if self.__image_infos[image_id].step != BEHIAVIOR_RECOGNIZE_COMPLETE:
            print(self.__image_infos[image_id].step)
            warnings.warn("步骤顺序错误", UserWarning)
            return {}
        return self.__image_infos[image_id].preds
    
    def get_image_info_by_image_id(self, image_id: int) -> ImageInfo:
        """获取图像信息

        Args:
            image_id (int): 图像id

        Returns:
            ImageInfo: 图像信息
        """
        if not self.check_image_id_exist(image_id):
            warnings.warn(f"image id {image_id} not found", UserWarning)
            return None
        return self.__image_infos[image_id]