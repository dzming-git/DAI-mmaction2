from generated.protos.target_tracking import target_tracking_pb2, target_tracking_pb2_grpc
import grpc
from typing import Dict, List

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
        request.onlyTheLatest = True
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
            # 直接访问最后一个bbox，前提是我们确信至少有一个bbox
            if bboxs:  # 确保bboxs列表不为空
                last_bbox = bboxs[-1]  # 获取最后一个bbox
                x1 = last_bbox.x1
                y1 = last_bbox.y1
                x2 = last_bbox.x2
                y2 = last_bbox.y2
                results[id] = [x1, y1, x2, y2]  # 直接将最后一个bbox作为结果存储
        return results
