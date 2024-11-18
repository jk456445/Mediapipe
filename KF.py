import math
import numpy as np
from scipy.linalg import sqrtm
import os


class ukf:
    def __init__(self, f, h):
        self.f = f
        self.h = h
        self.Q = None
        self.R = None
        self.P = None
        self.x = None
        self.Z = None
        self.n = None
        self.m = None


import numpy as np
import cv2
import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(write, file):
    output_file = file
    write_list = write
     # 寫入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')


def read_all(folderpath):
    data = read_json(folderpath)


#一般卡爾曼
def Kalman_3D(data, id): #只輸入現在的keypoints
    # 初始化Kalman Filter
    kalman = cv2.KalmanFilter(3, 3)
    kalman.measurementMatrix = np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],], dtype=np.float32)  # 測量矩陣
    kalman.transitionMatrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], dtype=np.float32)  # 狀態轉移矩陣
    kalman.processNoiseCov = np.eye(3, dtype=np.float32) * 1  # 狀態過程噪聲協方差矩陣
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.01  # 測量噪聲協方差矩陣

    # 讀取數據
    
    #kalman.statePre = np.array([data[0]["keypoints"][id][1]*100, data[0]["keypoints"][id][2]*100, data[0]["keypoints"][id][3]*100,0,0,0], dtype=np.float32)
    kalman.statePost = np.array([data[0]["keypoints"][id][1], data[0]["keypoints"][id][2], data[0]["keypoints"][id][3],0,0,0], dtype=np.float32)
    #print("stat",kalman.statePost )


    # 初始化Kalman Filter的初始狀態
    

    # 預測和校正
    predictions = []
    
    for i in range(len(data)):
        #print("frame:", i+1)
        #print("id", id)

        # 校正

        measurement = np.array([data[i]["keypoints"][id][1],data[i]["keypoints"][id][2],data[i]["keypoints"][id][3]], dtype=np.float32)
        #print("mea",measurement)
        low=kalman.correct(measurement)
        #print("corr", low)
        # 預測下一個狀態
        corrected = kalman.predict()
        #print("pre",corrected)
        predictions.append(corrected[:3])  # 只保留預測的x、y、z坐標
        
    #print("predict:", predictions)
    return predictions #這種跑出來的第一幀不知道為什麼都會被correct成0,0,0，試很久都沒救


def _main(data):
    json_data = read_json(data)
    write_list =[] #要寫入json的東西
    temp = []
    #print(json_data[2])
    for id in range(33):
        #print(id)
        predict = Kalman_3D(json_data,id) #讀特定關節的每一幀，一個一個關節去預測
        temp.append(predict) #存下來，變成是依關節存，所以之後要換
    #print(temp[0][0])
    #print(temp[0][0]*100)
    #print("-----")
    #print(temp[32]/100)
    #print("-----")
    #print(temp[32][33])

    new_json = data.split(".")[0]+"_KF.json"
    #new_json = os.path.join(os.path.dirname(data),"kal.json")   
    for i in range(len(json_data)):
        this_dict = {}
        this_dict['frame'] =i+1
        for_fey = []
        for id in range(33):#哪個關節的第幾幀
            arr = (temp[id][i]).tolist()
            flat_list = [item[0] for item in arr]
            flat_list.insert(0,id) #插入id
            for_fey.append(flat_list)#換回去原本的公尺

        #print(for_fey)
        #print(len(for_fey))
        this_dict['keypoints'] = for_fey
        write_list.append(this_dict)
    #print(write_list)
    write_json(write_list,new_json) #這個json就沒有信心值了

_main("E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900_ch.json")
    # # 繪製預測結果
    # img = cv2.imread(picpath)
    # for i in range(len(predictions) - 1):
    #     pt1 = (int(predictions[i][0]), int(predictions[i][1]))
    #     pt2 = (int(predictions[i + 1][0]), int(predictions[i + 1][1]))
    #     color = (0, 255, 0)
    #     thickness = 2
    #     line_type = cv2.LINE_AA
    #     cv2.line(img, pt1, pt2, color, thickness, line_type)
    #     cv2.circle(img, pt1, 5, (0, 0, 255), 1)

    # # 保存圖像
    # cv2.imwrite('predicted_trajectory.jpg', img)

