import pickle
from os.path import exists
import os  
import cv2
import numpy as np 
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO  
import torch  
from api.deepsort import DeepSORTTracker
from api.yolo import YOLOPersonDetector
 
# constants
YOLO_MODEL = "./checkpoints/yolov7x.pt"
REID_MODEL = "./checkpoints/ReID.pb"
MAX_COS_DIST = 0.5
MAX_TRACK_AGE = 100 
  
  
def get_video_path():
    '''
    把视频路径添加到txt文件里
    '''
    # 设置视频文件的扩展名，例如 .mp4, .avi 等  
    video_extensions = ['.mp4', '.avi', '.mov']  
    
    # 定义data文件夹的路径和输出txt文件的路径  
    data_folder_path = 'data/JAAD_clips'  
    output_txt_path = 'data/video_paths.txt'  
    
    # 打开或创建用于写入视频路径的txt文件  
    with open(output_txt_path, 'w') as txt_file:  
        # 遍历data文件夹  
        for root, dirs, files in os.walk(data_folder_path):  
            for file in files:  
                # 检查文件是否是视频文件（基于扩展名）  
                if os.path.splitext(file)[1].lower() in video_extensions:  
                    # 拼接完整的视频文件路径  
                    video_path = os.path.join(root, file)  
                    # 将视频路径写入到txt文件中  
                    txt_file.write(video_path + '\n')  
 
def load_pickle():
    '''
    jaad_database.pkl是jaad数据集自带的类函数JAAD生成的数据,具体详情其内部函数generate_database
    '''
    cache_file = "./data/jaad_database.pkl"
    if exists(cache_file):
        with open(cache_file, 'rb') as fid:
            try:
                database = pickle.load(fid)
            except:
                database = pickle.load(fid, encoding='bytes')
    return database
    
def show_frame(frame):
    '''
    展示帧图片
    '''
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    # 使用matplotlib来显示图像  
    plt.imshow(frame_rgb)  
    plt.axis('off')  # 可选，关闭坐标轴  
    plt.show()  # 显示图像
    
def read_video_path(path): 
    '''
    读取图片路径
    ''' 
    # 打开txt文件并读取内容  
    with open(path, 'r') as file:  
        lines = file.readlines()  
    # 去除每行末尾的换行符（如果有的话）  
    video_paths = [line.strip() for line in lines]  
    return video_paths

def get_video_name(file_path):
    '''
    获取视频名称
    '''
    # 使用os.path.basename获取文件名（包括扩展名）  
    file_name = os.path.basename(file_path)  
    # 使用os.path.splitext去除扩展名  
    video_name = os.path.splitext(file_name)[0]  
    return video_name

def calculate_iou(bbox1, bbox2):  
    """  
    计算两个bbox之间的IOU值。  
    bbox格式为[x_min, y_min, x_max, y_max]。  
    """  
    x1, y1, x2, y2 = bbox1  
    x_a, y_a, x_b, y_b = bbox2  
  
    # 计算交集区域的坐标  
    xi1 = max(x1, x_a)  
    yi1 = max(y1, y_a)  
    xi2 = min(x2, x_b)  
    yi2 = min(y2, y_b)  
  
    # 计算交集区域的宽度和高度  
    inter_width = max(xi2 - xi1, 0)  
    inter_height = max(yi2 - yi1, 0)  
  
    # 计算交集区域的面积  
    intersection = inter_width * inter_height  
  
    # 计算两个bbox的面积  
    area1 = (x2 - x1) * (y2 - y1)  
    area2 = (x_b - x_a) * (y_b - y_a)  
  
    # 计算并集区域的面积  
    union = area1 + area2 - intersection  
  
    # 避免除以0  
    if union == 0:  
        return 0  
  
    # 计算IOU  
    iou = intersection / union  
    return iou 

def match_bboxes(database_annotations, detected_bboxes, iou_threshold=0.5):  
    """  
    通过IOU匹配database中的annotations和检测到的bboxes。  
    返回匹配对的列表。  
    """  
    matched_pairs = []  
    log = [] 
    matched_track_ids = set()
  
    for db_id,data in database_annotations.items():
        best_match_id = None  
        best_iou = -1   
        for track_id, track_bboxes in detected_bboxes.items():
            # 检查track_id是否已经被匹配过  
            if track_id in matched_track_ids:  
                continue  # 跳过已匹配的track_id
            cnt = 0
            iou = 0
            for db_bbox in data['bbox']:  
                if cnt>=len(track_bboxes):
                    break
                iou += calculate_iou(db_bbox, track_bboxes[cnt]) 
                cnt += 1
            iou_avg = iou / len(track_bboxes) 
            if iou_avg > best_iou:  
                best_match_id = track_id  
                best_iou = iou_avg   
        if best_match_id == None:
            return matched_pairs,log
        elif best_iou > iou_threshold:  
            matched_pairs.append((db_id, best_match_id, best_iou))  
            matched_track_ids.add(best_match_id) 
        else:
            log.append((best_iou,{'yolo_bbox':detected_bboxes[best_match_id][0],'jaad_bbox':data['bbox'][0]}))
            matched_pairs.append((db_id, best_match_id, best_iou))
    return matched_pairs,log 

def match_people(input_vid,database):
    '''
    通过自己行人id匹配jaad数据集的行人id
    '''  
    detector = YOLOPersonDetector()  
    detector.load(YOLO_MODEL)  
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)  
    video = cv2.VideoCapture(input_vid)  
    people_bbox_dict = {}  # 用于存储人物ID和对应的前五个bbox  
    
    detected_ids = set()  # 用于存储检测到的唯一ID集合
    while True:  
        ret, frame = video.read()  
        if not ret:  
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        detections = detector.detect(frame)  
        try:  
            bboxes, scores, _ = np.hsplit(detections, [4, 5])  
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  
        except ValueError:  
            bboxes = np.empty(0)  
  
        # 如果有检测结果，进行追踪  
        if bboxes.size > 0:  
            people = tracker.track(frame, bboxes, scores.flatten())  
            for person in people:  
                bbox = person["bbox"]  
                track_id = person["track_id"]  
                detected_ids.add(track_id)  # 将检测到的ID添加到集合中 
                # 如果跟踪ID不在字典中，则添加它并初始化bbox列表  
                if track_id not in people_bbox_dict:  
                    people_bbox_dict[track_id] = []  
  
                if len(people_bbox_dict[track_id]) > 10:  
                    continue
                # 将bbox添加到对应跟踪ID的列表中，并保留前10个  
                people_bbox_dict[track_id].append(bbox)  

    matched_pairs,log=match_bboxes(database[get_video_name(input_vid)]["ped_annotations"],people_bbox_dict)
    # 将matched_pairs写入JSON文件
    path = f'./data/match_data/{get_video_name(input_vid)}'
    if not os.path.exists(path):  
        os.makedirs(path)   
    with open(f'{path}/matched_pairs.json', 'w') as f:  
        json.dump(matched_pairs, f, ensure_ascii=False, indent=4)  
    
    # 将log写入JSON文件 
    with open(f'{path}/log.json', 'w') as f:  
        json.dump(log, f, ensure_ascii=False, indent=4)
        
    video.release()  
    cv2.destroyAllWindows()   
    
def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    """
    This function returns a VideoWriter object with the same codec as the input VideoCapture object
    """
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))

def extract_skeleton_with_yolov8(img_bgr,ori_img,model,ox,oy):
    '''
    通过yolov8提取人体骨架关键点
    '''
    kpt_color_map = {
        0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': 6},  # 鼻尖
        1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': 6},  # 右边眼睛
        2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': 6},  # 左边眼睛
        3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': 6},  # 右边耳朵
        4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': 6},  # 左边耳朵
        5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 右边肩膀
        6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 左边肩膀
        7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': 6},  # 右侧胳膊肘
        8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': 6},  # 左侧胳膊肘
        9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': 6},  # 右侧手腕
        10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': 6},  # 左侧手腕
        11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': 6},  # 右侧胯
        12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': 6},  # 左侧胯
        13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': 6},  # 右侧膝盖
        14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': 6},  # 左侧膝盖
        15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': 6},  # 右侧脚踝
        16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': 6},  # 左侧脚踝
    }
    skeleton_map = [
        {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 1},  # 右侧脚踝-右侧膝盖
        {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 1},  # 右侧膝盖-右侧胯
        {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 1},  # 左侧脚踝-左侧膝盖
        {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 1},  # 左侧膝盖-左侧胯
        {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 1},  # 右侧胯-左侧胯
        {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 1},  # 右边肩膀-右侧胯
        {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 1},  # 左边肩膀-左侧胯
        {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 1},  # 右边肩膀-左边肩膀
        {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 1},  # 右边肩膀-右侧胳膊肘
        {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 1},  # 左边肩膀-左侧胳膊肘
        {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 1},  # 右侧胳膊肘-右侧手腕
        {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 1},  # 左侧胳膊肘-左侧手腕
        {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 1},  # 右边眼睛-左边眼睛
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 1},  # 鼻尖-左边眼睛
        {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 1},  # 鼻尖-左边眼睛
        {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 1},  # 右边眼睛-右边耳朵
        {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 1},  # 左边眼睛-左边耳朵
        {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 1},  # 右边耳朵-右边肩膀
        {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 1}  # 左边耳朵-左边肩膀
    ]
    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    if bboxes_keypoints.size==0:
        #print(num_bbox)
        return ori_img,bboxes_keypoints
    for bbox_keypoints in bboxes_keypoints:
        bbox_keypoints[:, 0] += ox  
        bbox_keypoints[:, 1] += oy 

    for idx in range(num_bbox):  # 遍历每个框
        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = round(bbox_keypoints[srt_kpt_id][0])
            srt_kpt_y = round(bbox_keypoints[srt_kpt_id][1])
            srt_kpt_conf = bbox_keypoints[srt_kpt_id][2]  # 获取起始点置信度
            # print(srt_kpt_conf)
            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = round(bbox_keypoints[dst_kpt_id][0])
            dst_kpt_y = round(bbox_keypoints[dst_kpt_id][1])
            dst_kpt_conf = bbox_keypoints[dst_kpt_id][2]  # 获取终止点置信度
            # print(dst_kpt_conf)
            # 获取骨架连接颜色
            skeleton_color = skeleton['color']
            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']
            # 如果起始点和终止点的置信度都高于阈值，才画骨架连接
            if srt_kpt_conf > 0.5 and dst_kpt_conf > 0.5:
                # 画骨架连接
                ori_img = cv2.line(ori_img, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                                   thickness=skeleton_thickness)
        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = round(bbox_keypoints[kpt_id][0])
            kpt_y = round(bbox_keypoints[kpt_id][1])
            kpt_conf = bbox_keypoints[kpt_id][2]  # 获取该关键点置信度
            if kpt_conf > 0.5:
                # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                ori_img = cv2.circle(ori_img, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
    return ori_img,bboxes_keypoints

def track_people_and_save_frames(input_vid, save_path): 
    '''
    用yolov7提取人体边框,用yolov8提取行人骨架关键点,并把视频和骨架数据保存下来
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('Models/yolov8n-pose.pt')
    model.to(device)
    #net = cv2.dnn.readNetFromTensorflow("graph_opt.pb") 
    detector = YOLOPersonDetector()  
    detector.load(YOLO_MODEL)  
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)  
    video = cv2.VideoCapture(input_vid)   
    output = video_writer_same_codec(video, save_path)
    skeletons_dict = {}
    frame_count = 0
    while True:  
        ret, frame = video.read() 
        if not ret:  
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        detections = detector.detect(frame)  
        try:  
            bboxes, scores, _ = np.hsplit(detections, [4, 5])  
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  
        except ValueError:  
            bboxes = np.empty(0)  
        if bboxes.size > 0:  
            people = tracker.track(frame, bboxes, scores.flatten())  
            for person in people: 
                bbox = person["bbox"]
                x1, y1, x2, y2 = bbox  
                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.shape[0] <= 0 or cropped_frame.shape[1] <= 0:
                    continue
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                skeleton = []
                frame,skeleton = extract_skeleton_with_yolov8(cropped_frame,frame,model,x1,y1)
                if skeleton.size==0:
                    continue
                track_id = person["track_id"]  
                if track_id not in skeletons_dict:  
                    skeletons_dict[track_id] = []  
                #skeletons_dict[track_id].append(skeleton[0])  
                skeletons_dict[track_id].append({'skeleton': skeleton[0], 'frame_number': frame_count}) 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)  
        frame_count += 1
    for track_id, skeletons in skeletons_dict.items():   
        skeleton_dir = f"skeleton/{get_video_name(input_vid)}/"  # 创建文件夹路径
        # 如果文件夹不存在，则创建文件夹  
        if not os.path.exists(skeleton_dir):  
            os.makedirs(skeleton_dir)  
        skeleton_path = os.path.join(skeleton_dir, f"data_{track_id}.npy")  # 创建文件完整路径   
        # 将骨架数据写入文件  
        np.save(skeleton_path, skeletons) 
    video.release()  
    output.release()  
    cv2.destroyAllWindows()
    return skeletons_dict

def make_lable(input_vid,database): 
    '''
    制作标签
    input_vid:输入的视频路径
    database:包含jaad一个视频每个人的数据也就是database['video_xxxx']["ped_annotations"]里的数据
    '''
    try:  
        with open(f'data/match_data/{get_video_name(input_vid)}/matched_pairs.json', 'r') as file:  
            data = json.load(file)  
    except FileNotFoundError:  
        print(f"file {get_video_name(input_vid)}/matched_pairs.json not found")  
        exit(1)  
    except json.JSONDecodeError:  
        print(f"file {get_video_name(input_vid)}/matched_pairs.json not json")  
        exit(1)  
    
    # 初始化一个列表来保存大于0.5阈值的ID  
    ids_above_threshold = []  
    
    # 遍历JSON数组中的每个子数组  
    for item in data:  
        # item的第三个元素是阈值  
        threshold = item[2]  
        if threshold > 0.5 and item[0].endswith('b'):  
            # 如果阈值大于0.5且id含b，则将前两个元素（ID）添加到列表中  
            ids_above_threshold.extend(item[:2])  

    # 根据ids_above_threshold列表的内容  
    if ids_above_threshold:  
        for id_pair in [id_pair for i in range(0, len(ids_above_threshold), 2) for id_pair in [tuple(ids_above_threshold[i:i+2])]]: 
            # 提取 frames、occlusion 和 behavior  
            frames_list = database[id_pair[0]]['frames']  
            occlusion_list = database[id_pair[0]]['occlusion']  
            behavior_list = database[id_pair[0]]['behavior'] 
            # 初始化一个空列表来保存匹配的结果  
            matched_items = []  
            
            skeleton_ = np.load(f'skeleton/{get_video_name(input_vid)}/data_{id_pair[1]}.npy', allow_pickle=True)
            # 遍历 skeleton_data 中的每个条目  
            for skel_dict in skeleton_:  
                frame = skel_dict['frame_number']  
                # 查找 frames_list 中与当前 frame 相等的索引  
                frame_index = frames_list.index(frame) if frame in frames_list else None  
                # 如果找到了相同的 frame  
                if frame_index is not None:  
                    # 提取对应的 occlusion 和 behavior  
                    occlusion = occlusion_list[frame_index]  
                    cross = behavior_list['cross'][frame_index]  
                    reaction = behavior_list['reaction'][frame_index] 
                    hand_gesture = behavior_list['hand_gesture'][frame_index] 
                    look = behavior_list['look'][frame_index] 
                    action = behavior_list['action'][frame_index] 
                    nod = behavior_list['nod'][frame_index] 
                    # 构建新的字典并添加到 matched_items 列表中  
                    matched_item = {  
                        'skleton': skel_dict['skeleton'],  
                        'occlusion': occlusion,  
                        'cross': cross,
                        'reaction': reaction,
                        'hand_gesture': hand_gesture,
                        'look': look,
                        'action': action,
                        'nod': nod
                    }  
                    matched_items.append(matched_item)  
            # 如果文件夹不存在，则创建文件夹  
            if not os.path.exists(f'data/final_data/{get_video_name(input_vid)}'):  
                os.makedirs(f'data/final_data/{get_video_name(input_vid)}') 
            # 保存匹配的结果到新的 numpy 文件中  
            np.save(f'data/final_data/{get_video_name(input_vid)}/data_{id_pair[1]}.npy', matched_items)
                       
    else:  
        # 如果没有ID满足条件，则输出文件名（不带后缀）   
        print(f"this file:{get_video_name(input_vid)}/matched_pairs.json threshold so low,please check")

def load_npy_test(filepath):
    # 使用NumPy的load函数来读取数据  
    skeleton_data = np.load(filepath, allow_pickle=True)  
    
    # 打印读取到的骨架数据
    print(len(skeleton_data))    
    print(skeleton_data) 

def make_intention(input_vid,database):
    '''
    制作意图标签
    input_vid:输入的视频路径
    database:包含jaad一个视频每个人的数据也就是database['video_xxxx']["ped_annotations"]里的数据
    '''
    try:  
        with open(f'data/match_data/{get_video_name(input_vid)}/matched_pairs.json', 'r') as file:  
            data = json.load(file)  
    except FileNotFoundError:  
        print(f"file {get_video_name(input_vid)}/matched_pairs.json not found")  
        exit(1)  
    except json.JSONDecodeError:  
        print(f"file {get_video_name(input_vid)}/matched_pairs.json not json")  
        exit(1)  
    
    # 初始化一个列表来保存大于0.5阈值的ID  
    ids_above_threshold = []  
    
    # 遍历JSON数组中的每个子数组  
    for item in data:  
        # item的第三个元素是阈值  
        threshold = item[2]  
        if threshold > 0.5 and item[0].endswith('b'):  
            # 如果阈值大于0.5且id含b，则将前两个元素（ID）添加到列表中  
            ids_above_threshold.extend(item[:2])  

    output_dir = f'data/intention_data/{get_video_name(input_vid)}'
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
    data_list = []
    # 根据ids_above_threshold列表的内容  
    if ids_above_threshold:   
        for id_pair in [id_pair for i in range(0, len(ids_above_threshold), 2) for id_pair in [tuple(ids_above_threshold[i:i+2])]]:  
            intention = database[id_pair[0]]['attributes']['crossing']  
            data_item = {f'data_{id_pair[1]}': intention}  
            data_list.append(data_item)
        
        output_file = os.path.join(output_dir, f'intention_data.json')
        with open(output_file, 'w') as f:  
            json.dump(data_list, f, ensure_ascii=False, indent=4) 
                        
    else:  
        # 如果没有ID满足条件，则输出文件名（不带后缀）   
        print(f"this file:{get_video_name(input_vid)}/matched_pairs.json threshold so low,please check")

def run():
    database = load_pickle()
    videos_path = read_video_path('data/video_paths.txt')
    for _vid in range(len(videos_path)):
        print(videos_path[_vid])
        match_people(videos_path[_vid],database)
    '''
    #下面代码用于获取行人骨架并把骨架图画在视频里
    videos_path = read_video_path('data/video_paths.txt')
    for _vid in range(len(videos_path)):
        print(videos_path[_vid])
        t=track_people_and_save_frames(videos_path[_vid],f'./data/processed_video/pro_{get_video_name(videos_path[_vid])}.mp4')
    '''
    '''
    #制作意图标签
    videos_path = read_video_path('data/video_paths.txt')
    for _vid in range(len(videos_path)):
        print(videos_path[_vid])
        make_intention(videos_path[_vid],database[get_video_name(videos_path[_vid])]["ped_annotations"])
    '''