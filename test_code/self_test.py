from test_code.wrapper_test import Mmaction2Recognizer

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

# human detection config file path (from mmdet)
builder.det_config = '/workspace/mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'

# human detection checkpoint file/url
builder.det_checkpoint = '/workspace/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

# the threshold of human action score
builder.action_score_thr = 0.4

# webcam id or input video file/url
builder.input_video = '/workspace/mmaction2/demo/test_video_structuralize.mp4'

# label map file
builder.label_map = '/workspace/mmaction2/tools/data/ava/label_map.txt'

# CPU/CUDA device option
builder.device = 'cuda:0'

# the fps of demo video output
builder.output_fps = 15

# the filename of output video
builder.out_filename = '/workspace/out.mp4'

# Whether to show results with cv2.imshow
builder.show = False

# Image height for human detector and draw frames
builder.display_height = 0

# Image width for human detector and draw frames
builder.display_width = 0

# give out a prediction per n frames
builder.predict_stepsize = 8

# Number of draw frames per clip
builder.clip_vis_length = 8
 
mmaction2_recognizer = builder.build()
mmaction2_recognizer.start()
