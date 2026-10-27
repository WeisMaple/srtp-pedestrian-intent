from DataProcess import *
from collections import deque 
from st_gcn import *

def predict_extract_skeleton(img_bgr,ori_img,model,ox,oy):
    '''
    通过yolov8提取人体骨架关键点
    '''
    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    if bboxes_keypoints.size==0:
        return ori_img,bboxes_keypoints
    for bbox_keypoints in bboxes_keypoints:
        bbox_keypoints[:, 0] += ox  
        bbox_keypoints[:, 1] += oy 
    return ori_img,bboxes_keypoints

def predict_draw(frame, predicted, bbox):
    x1, y1, x2, y2 = bbox  
    # 要绘制的文本
    if predicted == 0:
        text = f"pre:unclear intent"  
    elif predicted == 1:
        text = f"pre:not crossing" 
    elif predicted == 2:
        text = f"pre:crossing" 
    else:
        print('please check predicted')
        return 0
    # 设置文本的字体和颜色等属性  
    font = cv2.FONT_HERSHEY_SIMPLEX  
    font_scale = 0.5  
    font_color = (255, 255, 255)  # 白色文本  
    line_type = 2  
    thickness = 1  
    # 计算文本的尺寸  
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)  
    text_width, text_height = text_size  
    
    # 计算文本开始绘制的坐标，确保它在bbox的右上角内，并且不会超出图像边界  
    text_x = x2 - text_width - 5  # 从bbox的右边界减去文本宽度和一点额外的空间  
    text_y = y1 + 5  # 在bbox的上边界下方一点位置开始绘制  
    
    # 确保坐标不会超出图像边界  
    text_x = max(0, text_x)  
    text_y = max(0, text_y)  
    
    # 在图像上绘制文本  
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type, thickness)  

def video_with_classify(input_vid, save_path, net_model_path, num_class): 
    '''
    只是意图分类
    parameter:
        input_vid:视频路径
        save_path:视频保存路径
        net_model_path:分类模型路径
        num_class:种类数
    '''
    graph_args={
        "layout": 'yolov8',
        "strategy": 'spatial'}
    device = torch.device("cpu")
    YOLO_model = YOLO('Models/yolov8n-pose.pt')
    YOLO_model.to(device) 
    # 实例化模型  
    net_model = Model(3,num_class,graph_args,True)
    #加载模型训练
    model_state_dict = torch.load(net_model_path,map_location=torch.device('cpu'))  
    net_model.load_state_dict(model_state_dict) 
    net_model.eval()
    detector = YOLOPersonDetector()  
    detector.load(YOLO_MODEL)  
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)  
    video = cv2.VideoCapture(input_vid)   
    output = video_writer_same_codec(video, save_path)
    skeletons_dict = {}
    frame_count = 0
    flag = 0
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
                frame,skeleton = predict_extract_skeleton(cropped_frame,frame,YOLO_model,x1,y1)
                if skeleton.size==0:
                    continue
                track_id = person["track_id"]  
                # 如果track_id不在skeletons_dict中，初始化一个长度为50的deque  
                if track_id not in skeletons_dict:    
                    skeletons_dict[track_id] = deque(maxlen=50)  
                # 添加新骨架数据到deque中  
                skeletons_dict[track_id].append(skeleton[0])
                if flag == 0 and len(skeletons_dict[track_id]) == 50:
                    flag = 1
                    base = skeletons_dict[track_id]
                    base = np.stack(base, axis=0)
                    base = np.expand_dims(base, axis=0)
                    continue
                if len(skeletons_dict[track_id]) < 50:
                    continue
                else:
                    skeletons_3d = np.stack(skeletons_dict[track_id], axis=0) #(t,v,c)
                    skeletons_4d = np.expand_dims(skeletons_3d, axis=0) #(1,t,v,c)
                    skeletons_5d = []
                    skeletons_5d.append(base)
                    skeletons_5d.append(skeletons_4d)
                    skeletons_5d = np.array(skeletons_5d)
                    skeletons_5d = torch.from_numpy(skeletons_5d.astype(np.float32))
                    skeletons_5d = skeletons_5d.permute(0, 4, 2, 3, 1)
                    
                    mean = skeletons_5d.mean(dim=0, keepdim=True)  
                    std = skeletons_5d.std(dim=0, keepdim=True)#第一个维度为1，计算std会出现nan
                    std[std == 0] = 1e-6  
                    skeletons_5d = (skeletons_5d - mean) / std
                    with torch.no_grad():
                        outputs = net_model(skeletons_5d) 
                    _, predicted = torch.max(outputs, 1)  
                    predict_draw(frame, predicted[1], bbox)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)  
        frame_count += 1
    video.release()  
    output.release()  
    cv2.destroyAllWindows()

def data_pro(data,base,LSTM):
    '''
    根据50帧预测后50帧用来预测
    parameter:
        data:当前50帧的数据
        base:防止计算方差出现nan,加入前50帧和预测的后50帧计算std
        LSTM:LSTM模型
    return:st-gcn输入
    '''
    skeletons_3d = np.stack(data, axis=0) #(t,v,c)
    _mean = np.mean(skeletons_3d, axis=0)  
    _std = np.std(skeletons_3d, axis=0) + 1e-8
    data_list = deque(maxlen=50)
    for d in skeletons_3d:
        t = (d - _mean) / _std
        data_list.append(t)
    for i in range(5):
        pre_data = np.expand_dims(data_list, axis=0) #(batch,t,v,c)
        pre_data = torch.tensor(pre_data, dtype=torch.float32)
        pre_data = pre_data.view(pre_data.size(0), pre_data.size(1), 51)  # 形状变为 [batch, 50, 51]
        with torch.no_grad(): 
            outputs = LSTM(pre_data)
            outputs = np.array(outputs)
        output = outputs[0]
        for j in output:
            data_list.append(j)
    transformed_list = [(d * _std) + _mean for d in data_list]
    
    skeletons_3d = np.stack(transformed_list, axis=0)
    skeletons_4d = np.expand_dims(skeletons_3d, axis=0) #(1,t,v,c)
    skeletons_5d = []
    skeletons_5d.append(base)
    skeletons_5d.append(skeletons_4d)
    skeletons_5d = np.array(skeletons_5d)
    skeletons_5d = torch.from_numpy(skeletons_5d.astype(np.float32))
    skeletons_5d = skeletons_5d.permute(0, 4, 2, 3, 1)
                    
    mean = skeletons_5d.mean(dim=0, keepdim=True)  
    std = skeletons_5d.std(dim=0, keepdim=True)#第一个维度为1，计算std会出现nan
    std[std == 0] = 1e-6  
    skeletons_5d = (skeletons_5d - mean) / std
    return skeletons_5d

def predict_video(input_vid, save_path, net_model_path, LSTM_path, num_class): 
    '''
    预测
    parameter:
        input_vid:视频路径
        save_path:视频保存路径
        net_model_path:分类模型路径
        LSTM_path:LSTM模型路径
        num_class:种类数
    '''
    graph_args={
        "layout": 'yolov8',
        "strategy": 'spatial'}
    device = torch.device("cpu")
    YOLO_model = YOLO('Models/yolov8n-pose.pt')
    YOLO_model.to(device) 
    # 实例化模型  
    net_model = Model(3,num_class,graph_args,True).to(device) 
    #加载模型训练
    model_state_dict = torch.load(net_model_path,map_location=torch.device('cpu'))  
    net_model.load_state_dict(model_state_dict) 
    net_model.eval()
    LSTM = SkeletonLSTM(input_size=51, hidden_size=256, num_layers=10,  
                     output_size=51, output_frames=10).to(device) 
    model_state_dict = torch.load(LSTM_path,map_location=torch.device('cpu'))  
    LSTM.load_state_dict(model_state_dict) 
    LSTM.eval()
    
    detector = YOLOPersonDetector()  
    detector.load(YOLO_MODEL)  
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)  
    video = cv2.VideoCapture(input_vid)   
    output = video_writer_same_codec(video, save_path)
    skeletons_dict = {}
    frame_count = 0
    flag = 0
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
                frame,skeleton = predict_extract_skeleton(cropped_frame,frame,YOLO_model,x1,y1)
                if skeleton.size==0:
                    continue
                track_id = person["track_id"]  
                # 如果track_id不在skeletons_dict中，初始化一个长度为50的deque  
                if track_id not in skeletons_dict:    
                    skeletons_dict[track_id] = deque(maxlen=50)  
                # 添加新骨架数据到deque中  
                skeletons_dict[track_id].append(skeleton[0])
                if flag == 0 and len(skeletons_dict[track_id]) == 50:
                    flag = 1
                    base = skeletons_dict[track_id]
                    base = np.stack(base, axis=0)
                    base = np.expand_dims(base, axis=0)
                    continue
                if len(skeletons_dict[track_id]) < 50:
                    continue
                else:
                    skeletons_5d = data_pro(skeletons_dict[track_id],base,LSTM)
                    with torch.no_grad():
                        outputs = net_model(skeletons_5d) 
                    _, predicted = torch.max(outputs, 1)  
                    predict_draw(frame, predicted[1], bbox)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)  
        frame_count += 1
    video.release()  
    output.release()  
    cv2.destroyAllWindows()

def run():
    predict_video('data/JAAD_clips/video_0071.mp4','./data/result/predict71.mp4','Models/stgcn_gpu_pre.pth','Models/lstm_gpu_pre.pth',3)

