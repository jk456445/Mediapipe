import os
import cv2
import numpy as np
import mediapipe as mp
import json
from natsort import natsorted
import subprocess

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mppose = mp.solutions.pose

# def read_images_from_folder(folder_path):
#     image_files = sorted(os.listdir(folder_path))
#     print(image_files)
#     images = []
#     for file in image_files:
#         if file.endswith(".png") or file.endswith(".jpg"):
#             image_path = os.path.join(folder_path, file)
#             image = cv2.imread(image_path)
#             images.append(image)
#     return images

def find_pose_keypoints(folder_path,output): #讀資料夾裡面的圖片，輸出mediapipe座標
    #folder_path = folder_path+'/output'  #沒有做rembg的話用這個
    folder_path = folder_path   # optimize那邊直接讀傳過來的資料夾，裡面都是圖片
    pic_path = [os.path.join(folder_path,f) for f in os.listdir(folder_path)if f.endswith('.jpg')]
    pic_path = natsorted(pic_path)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    keypoints_list = []
    for i, file in enumerate(pic_path):
        print(file)
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.

        #print(pic_path.index(image))
        #frame_count = pic_path.index(image)
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
            image,        
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            this_dict = {} 
            keypoints = []
            for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                keee = list([id,landmark.x, landmark.y, landmark.z, landmark.visibility])
                keypoints.append(keee)
            this_dict['frame'] = i+1
            this_dict['keypoints'] =keypoints
            keypoints_list.append(this_dict)
            write_path = os.path.join(output,os.path.basename(file))
            cv2.imwrite(write_path, image)
    print(len(keypoints_list))
    holistic.close()
    return keypoints_list

def save_keypoints_to_json(keypoints_list, output_json_path):
    with open(output_json_path, 'w') as json_file:
        json.dump(keypoints_list, json_file)
    print("json file", output_json_path)

def for_video(readvideo):
    #video_name = os.path.splitext(os.path.basename(readvideo))[0]  # 影片的名字
    #print(video_name)
    folder = readvideo
    video_name = os.path.basename(folder)
    output_folder = f"./video/media/{video_name}/"
    output_json_path = f"./video/media/{video_name}/{video_name}.json"

    if os.path.isdir(output_folder):
        print(f"Delete old result folder: {output_folder}")
        subprocess.run(["rm", "-rf", output_folder])

    subprocess.run(["mkdir", output_folder])

    # images = read_images_from_folder(readvideo)
    keypoints_list = find_pose_keypoints(readvideo,output_folder)
    save_keypoints_to_json(keypoints_list, output_json_path)
    #print(keypoints_list)
    print("Finish processing images and save keypoints to JSON.")

    return output_json_path


if __name__=='__main__':
    #folder = "E://things/master/pose3d/video/01"
    #vid_path = [os.path.join(folder,f) for f in os.listdir(folder)if f.endswith('.mp4')]
    #print(vid_path)

    #folder = "E://things/master/pose3d/video/S001C001P001R001A027_rgb_sharpen"
    folder = "E://things/master/pose3d/video/myData0901-1/output/rembg"
    #folder = "E://things/master/pose3d/video/gait3d"
    return_path = for_video(folder)
