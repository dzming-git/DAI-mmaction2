from src.utils import singleton
from typing import Dict, Tuple, List
import queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.grpc.clients.target_tracking.target_tracking_client import TargetTrackingClient
from src.wrapper import Mmaction2Recognizer
from src.config.config import Config
import yaml
import traceback
import threading
import time

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
        
        self.image_harmony_address: List[str, str] = []
        self.image_harmony_client: ImageHarmonyClient = None
        self.loader_args_hash: int = 0  # image harmony中加载器的hash值
        
        self.target_tracking_address: List[str, str] = []
        self.target_tracking_client: TargetTrackingClient = None

        self.weights: str = ''
        self.device: str = ''
        self.image_id_queue: queue.Queue[int] = queue.Queue()
        self.recognizer: Mmaction2Recognizer = None
        self.stop_event = threading.Event()
        self.track_thread = None  # 用于跟踪线程的引用
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str]):
        if 'image harmony gRPC' == pre_service_name:
            self.image_harmony_address = [pre_service_ip, pre_service_port]
            self.image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            assert 'LoaderArgsHash' in args, 'arg: [LoaderArgsHash] not set'
            self.loader_args_hash = int(args['LoaderArgsHash'])
        if 'target tracking' == pre_service_name:
            self.target_tracking_address = [pre_service_ip, pre_service_port]
            self.target_tracking_client = TargetTrackingClient(pre_service_ip, pre_service_port, self.id)
            self.target_tracking_client.filter.add('person')
    
    def set_cur_service(self, args: Dict[str, str]):
        if 'Device' in args:
            self.device = args['Device']
        if 'Weight' in args:
            self.weight = args['Weight']
    
    def check(self) -> Tuple[bool, str]:
        try:
            assert self.target_tracking_client, 'Error: target_tracking_client not set.'
            assert self.image_harmony_client,    'Error: image_harmony_client not set.'
            assert self.loader_args_hash,        'Error: loader_args_hash not set.'
            assert self.weight,                  'Error: weight not set.'
            assert self.device,                  'Error: device not set.'
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            return False, error_info
        return True, 'OK'
    
    def start(self) -> Tuple[bool, str]:
        try:
            self.image_harmony_client.connect_image_loader(self.loader_args_hash)
            builder = Mmaction2Recognizer.Mmaction2Builder()
            config = Config()
            with open(config.weights_info, 'r') as f:
                weights_info = yaml.safe_load(f)
            assert self.weight in weights_info, f'Error: weight {self.weight} does not exist'
            weight_info = weights_info[self.weight]
            builder.config = weight_info['config']
            builder.checkpoint = weight_info['checkpoint']
            builder.label_map = weight_info['label_map']
            builder.device = self.device
            self.recognizer = builder.build()
            assert self.recognizer.load_model(), 'Error: recognizer.load_model() failed'
            self.stop_event.clear()  # 确保开始时事件是清除状态
            self.track_thread = threading.Thread(target=self.auto_recognize)
            self.track_thread.start()
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            return False, error_info
        return True, 'OK'

    def auto_recognize(self):
        loop_counter = 0  # 初始化循环计数器
        with open('timing_info.txt', 'w') as f:  # 打开文件用于写入
            while not self.stop_event.is_set():
                try:
                    start_time = time.time()
                    image_id_in_queue = self.image_id_queue.get(timeout=1)
                    get_queue_time = time.time() - start_time
                    f.write(f"Loop {loop_counter}: Getting image_id from queue: {get_queue_time:.5f} seconds\n")
                except queue.Empty:
                    continue
                
                loop_counter += 1  # 每次循环时增加计数器
                
                start_time = time.time()
                width, height = self.image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
                get_image_size_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Getting image size: {get_image_size_time:.5f} seconds\n")
                if 0 == width or 0 == height:
                    continue
                
                start_time = time.time()
                new_width, new_height = calculate_scaled_size(width, height)
                image_id, image = self.image_harmony_client.get_image_by_image_id(image_id_in_queue, new_width, new_height)
                get_image_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Getting and scaling image: {get_image_time:.5f} seconds\n")
                if 0 == image_id:
                    continue

                start_time = time.time()
                bboxes = self.target_tracking_client.get_result_by_image_id(image_id, True)
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
                get_bboxes_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Getting bboxes: {get_bboxes_time:.5f} seconds\n")
                if not self.recognizer.add_image_id(image_id):
                    continue
                if not self.recognizer.add_image(image_id, image):
                    continue
                if not self.recognizer.add_person_bboxes(image_id, bbox_per_person):
                    continue
                
                start_time = time.time()
                key_image_id = self.recognizer.get_key_image_id()
                get_key_image_id_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Getting key image id: {get_key_image_id_time:.5f} seconds\n")
                if 0 == key_image_id:
                    continue

                start_time = time.time()
                if not self.recognizer.predict_by_image_id(key_image_id):
                    continue
                predict_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Prediction: {predict_time:.5f} seconds\n")

                start_time = time.time()
                result = self.recognizer.get_result_by_image_id(key_image_id)
                get_result_time = time.time() - start_time
                f.write(f"Loop {loop_counter}: Getting result: {get_result_time:.5f} seconds\n")
                if len(result):
                    f.write(f"Loop {loop_counter}: image id: {image_id}, preds: {result}\n")

    def stop(self):
        self.stop_event.set()  # 设置事件，通知线程停止
        if self.track_thread:
            self.track_thread.join()  # 等待线程结束
        self.image_harmony_client.disconnect_image_loader()
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