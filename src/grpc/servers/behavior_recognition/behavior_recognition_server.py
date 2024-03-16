from generated.protos.behavior_recognition import behavior_recognition_pb2, behavior_recognition_pb2_grpc
import time
import traceback
from src.task_manager.task_manager import TaskManager
from src.config.config import Config

task_manager = TaskManager()
config = Config()

class BehaviorRecognitionServer(behavior_recognition_pb2_grpc.CommunicateServicer):
    def informImageId(self, request, context):
        """告知图像id

        Args:
            request (_type_): 请求
            context (_type_): 上下文
        
        Returns:
            _type_: 回应
        """
        response_code = 200
        response_message = ''
        try:
            task_id = request.taskId
            image_id = request.imageId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            recognizer = task_manager.tasks[task_id].recognizer
            image_id_exist = recognizer.check_image_id_exist(image_id)
            if not image_id_exist:
                task_manager.tasks[task_id].image_id_queue.put(image_id)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = behavior_recognition_pb2.InformImageIdResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def getResultByImageId(self, request, context):
        """根据图像id请求结果

        Args:
            request (_type_): 请求
            context (_type_): 上下文
        
        Returns:
            _type_: 回应
        """
        response_code = 200
        response_message = ''
        try:
            task_id = request.taskId
            image_id = request.imageId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            recognizer = task_manager.tasks[task_id].recognizer
            assert recognizer.check_image_id_exist(image_id), 'ERROR: The image ID does not exit.\n'   
            assert recognizer.get_step_by_image_id(image_id) == 0, 'ERROR: Image recognition incomplete.\n'  
            image_info = recognizer.get_image_info_by_image_id(image_id)
            assert image_info is not None,  'ERROR: Get image info failed.\n'  
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = behavior_recognition_pb2.GetResultByImageIdResponse()
        response.response.code = response_code
        response.response.message = response_message
        person_bboxes = image_info.origin_person_bboxes
        for person_id, label_confidences in image_info.preds.items():
            result_proto = response.results.add()
            for label, confidence in label_confidences:
                label_info_proto = result_proto.labelInfos.add()
                label_info_proto.label = label
                label_info_proto.confidence = confidence
            result_proto.personId = person_id
            person_bbox = person_bboxes[person_id]
            result_proto.x1 = person_bbox[0]
            result_proto.y1 = person_bbox[1]
            result_proto.x2 = person_bbox[2]
            result_proto.y2 = person_bbox[3]
        return response

    def getLatestResult(self, request, context):
        """获取最新结果: 最新的(key image)的识别结果 + 最新的bboxes(与key image不对应)

        Args:
            request (_type_): 请求
            context (_type_): 上下文

        Returns:
            _type_: 回应
        """
        response_code = 200
        response_message = ''
        try:
            task_id = request.taskId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            recognizer = task_manager.tasks[task_id].recognizer
            latest_predict_completed_image_id = recognizer.latest_predict_completed_image_id
            latest_add_image_id = recognizer.latest_add_image_id
            
            # 设置超时时间为 1 秒
            timeout = 1
            start_time = time.time()
            # 等待检测完成
            while recognizer.get_step_by_image_id(latest_add_image_id) != 0:
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("等待超时")
                time.sleep(0.01)
            
            latest_predict_completed_image_info = recognizer.get_image_info_by_image_id(latest_predict_completed_image_id)
            latest_add_image_info = recognizer.get_image_info_by_image_id(latest_add_image_id)
            assert latest_predict_completed_image_info is not None,  'ERROR: Get latest_predict_completed_image_info failed.\n'  
            assert latest_add_image_info is not None,  'ERROR: Get latest_add_image_info failed.\n'  
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = behavior_recognition_pb2.GetLatestResultResponse()
        response.response.code = response_code
        response.response.message = response_message
        person_bboxes = latest_add_image_info.origin_person_bboxes
        for person_id, label_confidences in latest_predict_completed_image_info.preds.items():
            result_proto = response.results.add()
            for label, confidence in label_confidences:
                label_info_proto = result_proto.labelInfos.add()
                label_info_proto.label = label
                label_info_proto.confidence = confidence
            result_proto.personId = person_id
            person_bbox = person_bboxes[person_id]
            result_proto.x1 = person_bbox[0]
            result_proto.y1 = person_bbox[1]
            result_proto.x2 = person_bbox[2]
            result_proto.y2 = person_bbox[3]
        return response
    
    def join_in_server(self, server):
        behavior_recognition_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
