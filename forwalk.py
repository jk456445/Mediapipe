import mediapipe as mp
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator 
import cv2
import csv
from PIL import Image
from sympy import*
import math
import sys
import time
from natsort import natsorted
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import os


#用來算步數的，感覺算步數用EMA比較好，不一定要加上UKF

def fourier2(x, a0, a1, b1, a2, b2, w):
    return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)

def stad_count(csv_name): #算次數用
    with open(csv_name, 'r') as file:
        # read csv
        reader = csv.reader(file)
        column2_data = [row[1] for row in reader]

    numpy_array = np.array(column2_data[1:], dtype=float) #把距離都取出來
    #print(numpy_array)

    y_data = numpy_array
    num_points = len(y_data)
    # y_data = y_data[start_index:end_index]
    # normalize [0, 1]
    y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
    # make X data
    x_data = np.arange(1, len(y_data)+1)

    std_list = []
    for i in range(1, 16):
        guess = i/100
        # 使用 curve_fit
        initial_guess = [0, 0, 0, 0, 0, guess]  # 初始值，老師說之後要用迴圈去調整
    
        fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=5000, method='lm')

        
        x_fit = x_data
        y_fit = fourier2(x_data, *fit_params)
        residuals = y_data - y_fit
        #print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
        #print("N:", str(len(residuals)))
        #print("dev:", np.sum(y_data))
        residuals_std = np.std(residuals)
        residuals_std = round(residuals_std,2)
        std_list.append(residuals_std)
        #print("std_list", std_list)
    
    min_std = np.argmin(std_list) #找最小值的index
    new_guess = (min_std+1)/100



    initial_guess = [0, 0, 0, 0, 0, new_guess]  # 初始值，老師說之後要用迴圈去調整

    fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=50000, method='lm')

    #print("標準差:", residuals_std)
    
   
    #print("参数:", fit_params)

    # 畫圖
    #plt.scatter(x_data, y_data, label='Data') #原本的距離跟len


    x_fit = x_data
    y_fit = fourier2(x_data, *fit_params)
    
    peaks, _ = find_peaks(y_data, prominence=0.1, distance = 10)  #prominence走路設了0.1，
    valleys, _ = find_peaks(-y_data, prominence=0.1, distance = 10)
    print("peak:",peaks)
    time = len(peaks)  #踏步只用peak就好 
    print("howmany",len(peaks)) #算次數 ，踏步的開始必須是站好，不然會出問題
    
    max_vals = []
    min_vals = []
    for i in peaks:
        #print(y_data[i])
        max_vals.append(y_data[i])
    for j in valleys:
        #print(y_data[i])
        min_vals.append(y_data[j])
    print('Max values:', max_vals)


    #plt.plot(x_fit, y_fit, 'r-', label='Fit')
    plt.plot(y_data)
    plt.xlabel('x')
    plt.ylabel('y')
    x_major_locator = MultipleLocator(10) #把x軸刻度設1, 存在變數裡
    ax = plt.gca() #ax為兩個座標軸的實例
    ax.xaxis.set_major_locator(x_major_locator) #把x座標軸刻度設為1的倍數
    plt.xlim(0.5,len(y_data))
    plt.scatter(peaks, max_vals, c ='red')
    # 顯示圖表
    #pic_num = csv_name.split(".")[0]+"_distpic"
    #print(pic_num)
    #cccc = pic_num
    #folder = "./"+cccc+"/test/"
    pic = csv_name.split(".")[0]+"_dist.png" #圖片存檔
    plt.savefig(pic)
    #plt.show()
    
    return time, peaks, valleys

def write_json(write, file):
    output_file = file
    write_list = write
     # 寫入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')

def readjson(path):
    with open(path, newline='') as jsonfile:
        data = json.load(jsonfile)
        # 或者這樣
        # data = json.loads(jsonfile.read())
        #print(data)
        return data
    
#計算兩個點之間的距離
def eucliDist(A, B):
    
    first = np.array(A)#只抓X,Y,Z軸
    #print(first)
    second = np.array(B)
    dist = np.sqrt(np.sum((first-second)**2))
    #print("dist:", dist)
    return dist


#計算角度用，這個是三點算角度的
def angle_in_space(A, B, C):#軒立的，對的，B是夾的角
    x1, y1, z1 = A[0], A[1], A[2]
    x2, y2, z2 = B[0], B[1], B[2]
    x3, y3, z3 = C[0], C[1], C[2]
    # 計算向量AB和BC
    AB = (x2 - x1, y2 - y1, z2 - z1)#把要夾的那個點都放前面
    BC = (x2 - x3, y2 - y3, z2 - z3)
    
    # 計算兩向量的點乘
    dot_product = AB[0] * BC[0] + AB[1] * BC[1] + AB[2] * BC[2]
    
    # 計算向量的模
    length_AB = math.sqrt(AB[0]**2 + AB[1]**2 + AB[2]**2)
    length_BC = math.sqrt(BC[0]**2 + BC[1]**2 + BC[2]**2)
    
    # 計算夾角的餘弦值
    cos_theta = dot_product / (length_AB * length_BC)
    
    # 計算夾角的弧度，並轉換為度
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    
    return round(theta_deg, 2)

def steplength(whvalley, data): #把valley的位置丟進來，因為這是我們要算步長的位置，現在試試都valley
    lastpoint = 0
    point = whvalley
    aver_step = []
    for j in range(len(point)+1):
        if j == len(point): #沒有下一個peak了
            frame = len(data) #所以直接讀到最後一幀
        else:
            frame = point[j]
        print("frame:",frame)
        #zstep = []
        step = []
        
        for i in range(lastpoint,frame ):
            left_ankle = data[i]['keypoints'][27][1:4] #左腳 
            right_ankle = data[i]['keypoints'][28][1:4] #右腳
            
            #step_lengthz = abs(left_ankle[2]-right_ankle[2]) #左腳y減右腳y
            step_length = eucliDist(left_ankle, right_ankle) #現在這個是抓這個範圍裏最大，也可能不是踏下去的時候
            #print("step_length:",step_length)
            #zstep.append(step_lengthz)
            step.append(step_length)
        
        #print("zstep:",max(zstep))
        #print("wherez", zstep.index(max(zstep))) #本來在試要純看z還是直接看距離
        print("step:",max(step))   #發現不管怎麼測，就算算的是歐式距離，兩腳的間距都不會超過30，不管我有沒有抓對幀，全程都不超過30
        #print(step)
        print(f"抓取範圍:{lastpoint}到{frame}")
        print("最大步位置", lastpoint + step.index(max(step)))
        aver_step.append(max(step))
        lastpoint = frame #更新上一個peak的位置
    print("平均步長:", round(sum(aver_step)/len(aver_step),2))


def steplength2(whpeak, data): #這個是直接算peak，抓peak那個位置時的步長
    lastpoint = 0
    point = whpeak
    aver_step = []
    aver_stepz = []
    for frame in point:
        # if j == len(point): #沒有下一個peak了
        #     frame = len(data) #所以直接讀到最後一幀
        # else:
        #     frame = point[j]
        print("frame:",frame)
        #zstep = []
        #step = []
        left_ankle = data[frame]['keypoints'][27][1:4] #左腳 
        right_ankle = data[frame]['keypoints'][28][1:4] #右腳
        step_lengthz = abs(left_ankle[2]-right_ankle[2])
        step_length = eucliDist(left_ankle, right_ankle) 

        print("當前步長:",step_length)
        print("當前z:",step_lengthz)
        aver_step.append(step_length)
        aver_stepz.append(step_lengthz)


        #lastpoint = frame #更新上一個peak的位置
    print("平均步長:", round(sum(aver_step)/len(aver_step),2))
    average_step = round(sum(aver_step)/len(aver_step),2)
    print("平均z:", round(sum(aver_stepz)/len(aver_stepz),2))

    return average_step




#用來找波裡面的最大角度，存下來，走路的裡面沒有用到
def forall_angle(whvalley,data,whichhand): #要求要告訴我是哪隻手在動
    start = 0
    angle_max = []
    where = []


    for t in range(len(whvalley)+1): #讀valley之間的數，ex:valley有4個，那就會有5個波鋒，所以才+1
        if t == len(whvalley): #沒有下一個谷了
            valle = len(data) #直接讀到最後一幀
        else:
            valle = whvalley[t] #谷就是這個值


        temp_ = []#暫存這個波中的所有角度



        for i in range(start,valle):#在這個波裡面找最大角度
            #print("i",i)
            if whichhand=="right":
                tangle = angle_in_space(data[i]['keypoints'][16][1:4],data[i]['keypoints'][12][1:4],
                                    [data[i]['keypoints'][12][1],0,data[i]['keypoints'][12][3]])

                #print("now right")
            elif whichhand=="left":
                tangle = angle_in_space(data[i]['keypoints'][15][1:4],data[i]['keypoints'][11][1:4],
                                    [data[i]['keypoints'][11][1],0,data[i]['keypoints'][11][3]])

                #print("now left")
            #print("angle",tangle)
            temp_.append(tangle)


        angle_max.append(max(temp_))

        
        where.append(temp_.index(max(temp_))+start)

        start = valle
        


    return angle_max,where #最大角度跟哪一幀最大




if __name__ =="__main__":
    #json_file = "E://things/master/pose3d/result/crawl0101/crawl0101.json"
    #json_file = "E://things/master/pose3d/video/media/camwalk/camwalk166250.json"  # 
    json_file = "E://things/master/pose3d/video/media/stepcount5/outputEMA2.json" #看起來用EMA或原本的比較準

    #hand = "left" #輸入是右手動還是左手動
    #hand = "right" #輸入是右手動還是左手動 算步數不用
    data = readjson(json_file)

    get_list = [25,26,27,28]
    origin = [data[0]['keypoints'][x][1:4] for x in get_list] #第一幀資料的xyz，因為如果把另一邊看不到的也放進去的話，抓peak有誤差
    # origin_angle = angle_in_space(data[0]['keypoints'][15][1:4],data[0]['keypoints'][11][1:4],
    #                             [data[0]['keypoints'][11][1],0,data[0]['keypoints'][11][3]])


    #print(origin)

dist_all3=[]
for frame in range(1,len(data)):
    get_list = [25,26,27,28]
    this_framekey = [data[frame]['keypoints'][x][1:4] for x in get_list] #第一幀資料的xyz，因為如果把另一邊看不到的也放進去的話，抓peak有誤差

    this_distance = eucliDist(origin,this_framekey)
    dist_all3.append(this_distance)
    #get=np.load(file)
    #np0 = np.load(np_sorted3[0])
    #gettt =np.empty((0, 4), dtype=np.float32)
    #np00 = np.empty((0, 4), dtype=np.float32)
    #np00 = np.vstack([np00, np0[13:23]])
    #np00 = np.vstack([np00, np0[27:]])
    #print(get.shape)
    # if get.shape == np0.shape:
    #     #print(file)
    #     gettt = np.vstack([gettt, get[13:23]]) #只存手跟腳
    #     gettt = np.vstack([gettt, get[27:]])
    #     #ske_dist3 = euclidean.eucliDist_no(np_sorted[0], file, 11 ,32)
    #     ske_dist3= np.sqrt(np.sum((np00-gettt)**2))
    #     #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
    #     dist_all3.append(ske_dist3)
#print("all:",dist_all3)

headers3 = ["pic", "dist"] #存3d

csv_name3 = json_file.split(".")[0]+"_dis.csv"
#path3 = csv_name3 + ".csv"
#print("path3",path3)
with open(csv_name3, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers3)
    for i in range(len(dist_all3)):
        writer.writerow([i,dist_all3[i]]) #存影片的動作所有距離


howmany, whpeak, whvalley = stad_count(csv_name3) #算總共做了幾次
print("times:",howmany)
print("whpeak:",whpeak)
print("whvalley:",whvalley)

average_step = steplength2(whpeak, data)   #用EMA的值，算步長跟步數的時候是最準的，這是算步長的


path = json_file.split('.')[0]+'walk.txt'
f = open(path, 'w')
f.write(f"幾步:{howmany}\n")
f.write(f"平均步長:{average_step}\n")

f.close()