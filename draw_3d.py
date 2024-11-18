import json
import matplotlib.pyplot as plt
import cv2
import os
import csv
import subprocess
import numpy as np
import math



#因為docker那邊不能畫圖，所以決定都用這邊畫，而且mediapipe自己畫都好慢(影片速度不對)，所以以後都用docker那邊找點，windows畫圖



LANDMARK_GROUP = [
    [8,6,5,4,0,1,2,3,7],#eyes
    [10,9], #mouth
    [11,13,15,17,19,15,21], #right arm
    [11,23,25,27,29,31,27], #right body side
    [12,14,16,18,20,16,22], #left arm
    [12,24,26,28,30,32,28], #left body side
    [11,12], #shoulder
    [23,24], #waist

]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


#這種畫法是畫線,也是3d
def  plot_world_landmarks(ax, landmarks, landmark_groups = LANDMARK_GROUP):
    if landmarks is None:
        return
    

    ax.cla()

    ax.set_xlim3d(-0.75,0.75)
    ax.set_ylim3d(-0.75,0.75)
    ax.set_zlim3d(1,-1)

    for group in landmark_groups:
        plotx, ploty, plotz = [], [], []

        for index in group:
            plotx.append(landmarks[index][0])
            ploty.append(landmarks[index][1])
            plotz.append(landmarks[index][2])
        # plotx = [landmarks.landmark[i].x for i in group]
        # ploty = [landmarks.landmark[i].y for i in group]
        # plotz = [landmarks.landmark[i].z for i in group]

        ax.plot(plotx, plotz, ploty)

    plt.pause(0.000000001)
    return

def read_json_vid(json_file):
    json_file_path = json_file

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    #jkey=list(json_data.keys())
    #print(jkey)
    #print(len(json_data))
    #print(json_data[0].keys())  #每個frame的key
    #print(json_data[0]['keypoints']) #第一幀的keypoints
    newww_list = [item[1:4] for item in json_data[0]['keypoints']]
    #print(newww_list)
    #print(len(newww_list))
    #print(json_data[0]) #第一幀的所有東西
    folder = os.path.dirname(json_file) #目錄名稱
    output_folder = folder+"/3dpic2/"
    if os.path.isdir(output_folder):
        print("Delete old result folder: {}".format(output_folder))
        
        subprocess.run(["rm", "-rf", output_folder])

    subprocess.run(["mkdir", output_folder])

    for i in range(len(json_data)):

        draw_key = [item[1:4] for item in json_data[i]['keypoints']]
        plot_world_landmarks(ax, draw_key) #畫線
        
        
        output_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        plt.savefig(output_path)


def main():
    read_json_vid("E://things/master/pose3d/video/media/faa/myData1201/myData1201EMA_UKFA_pic.json")
    #read_json_vid("E://things/master/pose3d/result/crawl0201/crawl0201.json")
    
    
if __name__=='__main__':
    main()

    