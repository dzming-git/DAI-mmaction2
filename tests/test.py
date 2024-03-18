from src.wrapper.mmaction2_recognizer import Mmaction2Recognizer
import cv2

def read_detections_from_txt(filepath):
    detections = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            # 先按照":"分隔帧号和检测结果
            frame_id, detection_str = line.strip().split(':', 1)
            frame_id = int(frame_id)
            # 将检测结果字符串转换为浮点数列表
            detection = list(map(float, detection_str.split(' ')))
            # 将检测结果添加到对应帧号的列表中
            if frame_id in detections:
                detections[frame_id].append(detection)
            else:
                detections[frame_id] = [detection]
    return detections

builder = Mmaction2Recognizer.Mmaction2Builder()

# # spatio temporal detection config file path
# builder.config = '/workspace/mmaction2/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'

# # spatio temporal detection checkpoint file/url
# builder.checkpoint = '/workspace/weights/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'

# spatio temporal detection config file path
builder.config = '/workspace/mmaction2/configs/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.py'

# spatio temporal detection checkpoint file/url
builder.checkpoint = '/workspace/weights/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth'

# the threshold of human detection score
builder.det_score_thr = 0.9

# the threshold of human action score
builder.action_score_thr = 0.4

# label map file
builder.label_map = '/workspace/mmaction2/tools/data/ava/label_map.txt'

# CPU/CUDA device option
builder.device = 'cuda:0'
 
recognizer = builder.build()
if not recognizer.load_model():
    exit()

cap = cv2.VideoCapture('/workspace/mmaction2/demo/test_video_structuralize.mp4')
bboxes = read_detections_from_txt('tests/test_video_structuralize_bboxes.txt')

image_id = 1

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break  # 如果没有帧了，就退出循环
    
    recognizer.add_image_id(image_id)
    recognizer.add_image(image_id, image)
    result = {}
    if image_id in bboxes:
        result = bboxes[image_id]
    person_bboxes = {}
    for person_id, r in enumerate(result):
        person_bboxes[person_id] = r
    recognizer.add_person_bboxes(image_id, person_bboxes)
    
    
    key_image_id = recognizer.get_key_image_id()
    if 0 != key_image_id:
        recognizer.predict_by_image_id(key_image_id)
        result = recognizer.get_result_by_image_id(key_image_id)
        print(result)
    image_id += 1
    