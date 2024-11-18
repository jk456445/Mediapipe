from ultralytics import YOLO
import ultralytics
import cv2
import math
import mediapipe as mp
import numpy as np
import os
os.environ['CUDA_VISIVLE_DIVICES']='0,1'
from natsort import natsorted
import subprocess
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import draw_plot
import subprocess
import json


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mppose = mp.solutions.pose


def write_json(write, file):
    output_file = file
    write_list = write
     # 寫入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')



def yolo_video():
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    ultralytics.checks()
    # Define path to video file




    #path = readvideo #輸入的影片

    read_video ='./video.myData0901-1.mp4'
    cap = cv2.VideoCapture(0) 
    path = './video/record.avi'
    print(path)
    #print(path.split('/'))
    video_name = (path.split('/')[2]).split('.')[0]  #video本身名字
    print(video_name)
    new_path = "./video/yolov8/"+video_name+"/"+video_name+'.avi' #新輸出的影片
    output_folder = "./video/yolov8/"+video_name+'/'
    if os.path.isdir(output_folder):
        print("Delete old result folder: {}".format(output_folder))
        
        subprocess.run(["rm", "-rf", output_folder])

    subprocess.run(["mkdir", output_folder])

    #print("create folder: {}".format(output_folder))
    #print("new:",new_path )


    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(new_path, fourcc, 20.0, (frame_width, frame_height))

    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]

    MARGIN = 30
    frame_count = 0
    write_list =[]

    while True:
        success, img = cap.read()
        
        results = model(img, stream=True)
        frame_count = frame_count+1
        
        if not success:
            break

        # coordinates
        for r in results:
            boxes = r.boxes #用於邊界框輸出的boxes對象
            #print(boxes)

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                #print("xyxy:",x1,",", y1,",", x2,",", y2 )
                #cv2.circle(img, (x2, y2), 5, (0,0,0), -1)長方形的右下角的點

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                #print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                #print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1] #左上角位置
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                #print("{}在{}".format(classNames[cls],org))
                #print("cls", cls) #person是0

                cv2.putText(img, classNames[cls]+str(confidence), org, font, fontScale, color, thickness)
                if cls==0 and confidence>=0.8:
                    with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as holistic:  #model_complexity=2就是兩個仁
                        #fig = plt.figure()
                        #ax = fig.add_subplot(111, projection="3d")
                        #fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
                        #下面的img的部分，是用slice來表示要抓的範圍，前面是height，後面是width
                        #results = holistic.process(cv2.cvtColor(img[y1-MARGIN:y2+MARGIN,x1-MARGIN:x2+MARGIN], cv2.COLOR_BGR2RGB))
                        results = holistic.process(cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))

                        if results.pose_landmarks:
                        #draw3d(plt, ax, results.pose_world_landmarks) #畫點
                        #plot_world_landmarks(ax, results.pose_world_landmarks ) #畫線
                        #draw_plot.plot_world_landmarks(plt,ax,results.pose_world_landmarks)
                        
                        #print("frame count:",frame_count)
                            use_keys = np.empty((0, 4), dtype=np.float32)
                            key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                            this_dict = {}  #用來存這一幀資料
                            #all3dkeys = np.empty((0, 4), dtype=np.float32)
                            key3 = []
                            mp_drawing.draw_landmarks(
                            img[y1-MARGIN:y2+MARGIN,x1-MARGIN:x2+MARGIN],
                            results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())
                            cv2.rectangle(img, (x1-MARGIN, y1-MARGIN), (x2+MARGIN, y2+MARGIN), (0, 255, 0), 3)
                            for id, landmark in enumerate(results.pose_landmarks.landmark):

                                use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                                use_keys = np.vstack([use_keys, use_key])


                                key_dist= np.vstack([key_dist, use_key])
                                #output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                                #np.save(output_file, key_dist)
                                #print("put 1")
                            for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                                #決定都用list存，不想用nparray了
                                
                                #use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                                keee = list([id,landmark.x, landmark.y, landmark.z, landmark.visibility])
                                #print(use_key)
                                #allkeys = np.vstack([allkeys, use_key])
                                #all3dkeys = np.vstack([all3dkeys, use_key])
                                key3.append(keee)
                            #print(key3)
                            #print("frame:",frame_count)
                            this_dict['frame'] = frame_count
                            this_dict['keypoints'] =key3
                            #print(this_dict['keypoints'])
                            write_list.append(this_dict)
                            #print("frame count:",frame_count)
        out.write(img)

        output_json = './video/yolov8/'+video_name+"/"+video_name+'.json'

        write_json(write_list,output_json)  #把keypoint都寫進json
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import time
    start = time.time()
    yolo_video()
    #yolo_video('./video/myData0302.mp4')


    end = time.time()
    print("執行時間:",end-start)