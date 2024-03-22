from src.utils import singleton
from typing import Dict, Tuple, List
import queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.grpc.clients.target_tracking.target_tracking_client import TargetTrackingClient
from src.wrapper import Mmaction2Recognizer
from src.config.config import Config
import yaml
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_scaled_size(width: int, height: int) -> Tuple[int, int]:
    max_width = 640
    max_height = 480

    # 如果原始尺寸小于规定尺寸，直接返回原始尺寸
    if width <= max_width and height <= max_height:
        return width, height

    # 计算宽度和高度的缩放比例
    width_ratio = max_width / width
    height_ratio = max_height / height

    # 选择较小的缩放比例，确保图像适应最大尺寸限制
    scale_ratio = min(width_ratio, height_ratio)

    # 计算缩放后的宽度和高度
    scaled_width = int(width * scale_ratio)
    scaled_height = int(height * scale_ratio)

    return scaled_width, scaled_height

class TaskInfo:
    def __init__(self, taskId: int):
        self.id: int = taskId
        
        self.__image_harmony_address: List[str, str] = []
        self.__image_harmony_client: ImageHarmonyClient = None
        self.__loader_args_hash: int = 0  # image harmony中加载器的hash值
        
        self.__target_tracking_address: List[str, str] = []
        self.__target_tracking_client: TargetTrackingClient = None

        self.__weight: str = ''
        self.__device_str: str = ''
        self.image_id_queue: queue.Queue[int] = queue.Queue()
        self.recognizer: Mmaction2Recognizer = None
        self.__stop_event = threading.Event()
        self.__track_thread = None  # 用于跟踪线程的引用
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str]):
        if 'image harmony gRPC' == pre_service_name:
            self.__image_harmony_address = [pre_service_ip, pre_service_port]
            self.__image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            if 'LoaderArgsHash' not in args:
                raise ValueError('Argument "LoaderArgsHash" is required but not set.')
            self.__loader_args_hash = int(args['LoaderArgsHash'])
        if 'target tracking' == pre_service_name:
            self.__target_tracking_address = [pre_service_ip, pre_service_port]
            self.__target_tracking_client = TargetTrackingClient(pre_service_ip, pre_service_port, self.id)
            self.__target_tracking_client.filter.add('person')
    
    def set_cur_service(self, args: Dict[str, str]):
        if 'Device' in args:
            self.__device_str = args['Device']
        if 'Weight' in args:
            self.__weight = args['Weight']
            config = Config()
            with open(config.weights_info, 'r') as f:
                weights_info = yaml.safe_load(f)
            if self.__weight not in weights_info:
                raise ValueError(f'Error: weight {self.__weight} does not exist')
    
    def check(self) -> None:
        if not self.__target_tracking_client:
            raise ValueError('Error: target_tracking_client not set.')
        if not self.__image_harmony_client:
            raise ValueError('Error: image_harmony_client not set.')
        if not self.__loader_args_hash:
            raise ValueError('Error: loader_args_hash not set.')
        if not self.__weight:
            raise ValueError('Error: weight not set.')
        if not self.__device_str:
            raise ValueError('Error: device not set.')
    
    def start(self) -> None:
        self.__image_harmony_client.connect_image_loader(self.__loader_args_hash)
        builder = Mmaction2Recognizer.Mmaction2Builder()
        config = Config()
        with open(config.weights_info, 'r') as f:
            weights_info = yaml.safe_load(f)
        weight_info = weights_info[self.__weight]
        builder.config = weight_info['config']
        builder.checkpoint = weight_info['checkpoint']
        builder.label_map = weight_info['label_map']
        builder.device = self.__device_str
        self.recognizer = builder.build()
        self.recognizer.load_model()
        self.__stop_event.clear()  # 确保开始时事件是清除状态
        self.__track_thread = threading.Thread(target=self.auto_recognize)
        self.__track_thread.start()

    def auto_recognize(self):
        while not self.__stop_event.is_set():
            try:
                image_id_in_queue = self.image_id_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                width, height = self.__image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
                if 0 == width or 0 == height:
                    continue
                
                new_width, new_height = calculate_scaled_size(width, height)
                image_id, image = self.__image_harmony_client.get_image_by_image_id(image_id_in_queue, new_width, new_height)
                if 0 == image_id:
                    continue

                bboxes = self.__target_tracking_client.get_result_by_image_id(image_id, True)
                bbox_per_person: Dict[int, List[float]] = {}
                for person_id, bbox in bboxes.items():
                    if 1 != len(bbox):
                        continue
                    bbox_per_person[person_id] = [
                        bbox[0].x1,
                        bbox[0].y1,
                        bbox[0].x2,
                        bbox[0].y2
                    ]
                if not self.recognizer.add_image_id(image_id):
                    continue
                if not self.recognizer.add_image(image_id, image):
                    continue
                if not self.recognizer.add_person_bboxes(image_id, bbox_per_person):
                    continue
                
                key_image_id = self.recognizer.get_key_image_id()
                if 0 == key_image_id:
                    continue

                if not self.recognizer.predict_by_image_id(key_image_id):
                    continue

                # result = self.recognizer.get_result_by_image_id(key_image_id)           
            except Exception as e:
                logging.error(e)


    def stop(self):
        self.__stop_event.set()  # 设置事件，通知线程停止
        if self.__track_thread:
            self.__track_thread.join()  # 等待线程结束
        self.__image_harmony_client.disconnect_image_loader()
        if self.recognizer:
            del self.recognizer  # 释放资源
        self.recognizer = None
                                                                                                                                         
@singleton
class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, TaskInfo] = {}
        self.__lock = threading.Lock()
    
    def stop_task(self, task_id: int):
        with self.__lock:
            if task_id in self.tasks:
                self.tasks[task_id].stop()
                del self.tasks[task_id]