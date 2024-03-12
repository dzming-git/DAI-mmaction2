from src.utils import temporary_change_dir
from src.wrapper.utils import *
from mmengine import Config
import time
import torch
import torch
import warnings
warnings.filterwarnings('always')

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
            
            # human detection config file path (from mmdet)
            self.det_config: str = ''
            
            # human detection checkpoint file/url
            self.det_checkpoint: str = ''
            
            # the threshold of human action score
            self.action_score_thr: float = 0.4
            
            # label map file
            self.label_map: str = ''
            
            # CPU/CUDA device option
            self.device: str = 'cuda:0'
            
            # give out a prediction per n frames
            self.predict_stepsize: int = 8
            
        def build(self):
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    warnings.warn("cuda is not available", UserWarning)
                self.device = 'cpu'
            return Mmaction2Recognizer(self)
    
    def __init__(self, builder: Mmaction2Builder):
        # 加载配置
        self.__config: Config = self.__load_config(builder.config)
        
        # 检测设备
        self.__device: str = builder.device
        
        # 目标检测
        self.__det_score_thr: float = builder.det_score_thr
        self.__det_config: str = builder.det_config
        self.__det_checkpoint: str = builder.det_checkpoint
        
        # 行为识别
        self.__action_score_thr: float = builder.action_score_thr
        self.__checkpoint: str = builder.checkpoint
        self.__label_map: str = builder.label_map
        
        # 预测步长
        self.__predict_stepsize: int = builder.predict_stepsize
        
        
        with temporary_change_dir('mmaction2'):
            self.__stdet_predictor = Mmaction2StdetPredictor(
                config=self.__config,
                checkpoint=self.__checkpoint,
                device=self.__device,
                score_thr=self.__action_score_thr,
                label_map_path=self.__label_map)
        
    def __load_config(self, config_path: str) -> Config:
        # init action detector
        config = Config.fromfile(config_path)

        try:
            # In our spatiotemporal detection demo, different actions should have
            # the same number of bboxes.
            config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
        except KeyError:
            pass
        return config
    
    def start(self):
        try:
            # Main thread main function contains:
            # 1) get data from read queue
            # 2) get human bboxes and stdet predictions
            # 3) draw stdet predictions and update task
            # 4) put task into display queue
            for able_to_read, task in self.__clip_helper:
                # get data from read queue

                if not able_to_read:
                    # read thread is dead and all tasks are processed
                    break

                if task is None:
                    # when no data in read queue, wait
                    time.sleep(0.01)
                    continue

                inference_start = time.time()

                # get human bboxes
                self.__human_detector.predict(task)

                # get stdet predictions
                self.__stdet_predictor.predict(task)

                # draw stdet predictions in raw frames
                self.__vis.draw_predictions(task)

                # add draw frames to display queue
                self.__clip_helper.display(task)

            # wait for display thread
            self.__clip_helper.join()
        except KeyboardInterrupt:
            pass
        finally:
            # close read & display thread, release all resources
            self.__clip_helper.clean()