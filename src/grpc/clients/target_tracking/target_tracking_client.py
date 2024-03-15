from generated.protos.target_tracking import target_tracking_pb2, target_tracking_pb2_grpc
import grpc
import cv2
from typing import Dict, Tuple, List
import numpy as np

class TargetTrackingClient:
    def __init__(self, ip:str, port: str, taskId: int):
        self.conn = grpc.insecure_channel(f'{ip}:{port}')
        self.client = target_tracking_pb2_grpc.CommunicateStub(channel=self.conn)
        self.task_id: int = taskId
        self.track_target_label: str = ''
        self.target_lable_id = -1
    
    def set_task_id(self, task_id: int):
        self.task_id = task_id
    
    # 只获取 label == 'person' 的结果
    def get_result_by_image_id(self, image_id: int) -> Dict[int, List[float]]:
        request = target_tracking_pb2.GetResultByImageIdRequest()
        request.taskId = self.task_id
        request.imageId = image_id
        request.wait = True
        response = self.client.getResultByImageId(request)
        if 200 != response.response.code:
            print(f'{response.response.code}: {response.response.message}')
            return {}
        results: Dict[int, List[float]] = {}
        for result in response.results:
            if result.label != 'person':
                continue
            id = result.id
            bboxs = result.bboxs
            results[id] = []
            for bbox in bboxs:
                x1 = bbox.x1
                y1 = bbox.y1
                x2 = bbox.x2
                y2 = bbox.y2
                results[id].append([x1, y1, x2, y2])
        return results
