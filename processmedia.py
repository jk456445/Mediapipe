import numpy as np
import statistics
import json
import csv

"""這是拿來再dataset中調整跳反邊的mediapipe"""



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
    

def eucliDist(A, B):
    
    first = np.array(A)#只抓X,Y,Z軸
    #print(first)
    second = np.array(B)
    dist = np.sqrt(np.sum((first-second)**2))
    #print("dist:", dist)
    return dist

def correct_or_wrong(ground_truth, compare): #正確數據，比較數據
    c = [ x for x in ground_truth if x not in compare] #  列出ground truth有，但是compare沒有的
    d = [ x for x in compare if x not in ground_truth]#  列出compare有，但是ground truth沒有的
    return c ,d



def get_direction(jsonfile):#沒用到
    data = readjson(jsonfile)
    data = data[2901:3501]
    front = []
    back = []
    notsure = []
    forward_state = []
    for frame in range(len(data)):
        pre_state = forward_state
        real_frame = frame+2901
        
        right_heel = data[frame]['keypoints'][30][1:4]
        right_foot = data[frame]['keypoints'][32][1:4]
        left_heel = data[frame]['keypoints'][29][1:4]
        left_foot = data[frame]['keypoints'][31][1:4]
        right_ankle = data[frame]['keypoints'][28][1:4]
        left_ankle = data[frame]['keypoints'][27][1:4]
        if left_ankle[2]>right_ankle[2] : #左腳z較大
            if left_foot[2]-left_heel[2]>=0.01:
                forward_state.append(str(False))
                back.append(real_frame)
            elif left_heel[2]-left_foot[2]>=0.01:
                forward_state.append(str(True))
                front.append(real_frame)
            elif left_foot[2]-left_heel[2]<0.01 and left_heel[2]-left_foot[2]<0.01:
                #forward_state = pre_state
                notsure.append(real_frame)
                                # 获取最后10个值
                last_10_values = forward_state[-10:]

                # 统计 True 和 False 的数量
                true_count = last_10_values.count("True")
                false_count = last_10_values.count("False")
                
                # 比较数量并输出结果
                if true_count > false_count and forward_state[-1]=="True":
                    front.append(real_frame)
                    
                elif true_count < false_count and forward_state[-1]=="False":
                    back.append(real_frame)
                    forward_state.append(str(False))
                else:
                    if forward_state[-1]=="True":
                        front.append(real_frame)
                    else:
                        back.append(real_frame)
                        forward_state.append(str(False))
                #if pre_state == True:
                    #front.append(real_frame)
                #else:
                    #back.append(real_frame)

        else: #右腳z較大
            if right_foot[2]-right_heel[2]>=0.01:
                forward_state.append(str(False))
                back.append(real_frame)
            elif right_heel[2]-right_foot[2]>=0.01:
                forward_state.append(str(True))
                front.append(real_frame)
            elif right_foot[2]-right_heel[2]<0.01 and right_heel[2]-right_foot[2]<0.01:
                #forward_state = pre_state
                notsure.append(real_frame)
                last_10_values = forward_state[-10:]

                # 统计 True 和 False 的数量
                true_count = last_10_values.count("True")
                false_count = last_10_values.count("False")
                
                # 比较数量并输出结果
                if true_count > false_count and forward_state[-1]=="True":
                    front.append(real_frame)
                    forward_state.append(str(True))
                    
                elif true_count < false_count and forward_state[-1]=="False":
                    back.append(real_frame)
                    forward_state.append(str(False))
                else:
                    if forward_state[-1]=="True":
                        front.append(real_frame)
                        forward_state.append(str(True))
                    else:
                        back.append(real_frame)
                        forward_state.append(str(False))
                # if pre_state == True:
                #     front.append(real_frame)
                # else:
                #     back.append(real_frame)

    
    #print("front", front)
    #print("back", back)
    #print("not sure", notsure)

    #製造ground truth
    gt_back = []
    for i in range(2901,3269):
        gt_back.append(i)
    gt_front = []
    for i in range(3290,3501):
        gt_front.append(i)
    side = []
    for i in range(3269,3290):
        side.append(i)

    front = [ x for x in front if x not in side]
    back = [ x for x in back if x not in side]
    wrong_front,toomuch_fr = correct_or_wrong(gt_front,front)
    print("wrong_front",wrong_front)
    print("front多抓",toomuch_fr)
    wrong_back,toomuch_ba = correct_or_wrong(gt_back,back)
    print("wrong_back",len(wrong_back))
    print("back多抓",toomuch_ba)






#用來改亂跳的數值，
def process_change(jsondata,get_thres): #要改的json data，速度list, 速度是從哪一幀開始哪一幀結束
    print(len(jsondata))
    data = jsondata
    change_data = []
    jump_up = []
    jump_down = []
    change_data.append(data[0])
    change_data.append(data[1])

    # change_data.append(data[3176])
    # change_data.append(data[3177])
    # change_data.append(data[3178])
    for frame in range(2,len(data)): #先在這個範圍測，然後改成把數值存去另一邊，才不會搞混
        print("frame",frame+2900)
        #print("frame",frame+1900)
        temp_dic = {'frame':frame+2901, 'keypoints':[]}
        #temp_dic = {'frame':frame+1901, 'keypoints':[]}
        this_frame_velocity = [] 
        dis_x = []
        dis_z = []
        for i in range(33):
            #print(i)
            #pre_position = change_data[-2]['keypoints'][i][1:4] #倒數第二個
            pre_position = change_data[-1]['keypoints'][i][1:4] #用倒數一幀來比較有沒有跳
            #print("pre",pre_position)
            now_position = data[frame]['keypoints'][i][1:4]
            #print("now",now_position)
            v= eucliDist(pre_position, now_position)
            #print("v", v)

            this_frame_velocity.append(v)
            x_dis = abs(data[frame]['keypoints'][i][1]-change_data[-2]['keypoints'][i][1])
            z_dis = abs(data[frame]['keypoints'][i][3]-change_data[-1]['keypoints'][i][3])
            dis_x.append(x_dis)
            dis_z.append(z_dis)
        new_keypoints = []
        
        #pre_rishoulder = change_data[-2]['keypoints'][12][1:4]
        #pre_leshoulder = change_data[-2]['keypoints'][11][1:4]
        #正面的時候11的x>12的x,背面的時候相反
        #bi_x_dis = change_data[-2]['keypoints'][11][1]-change_data[-2]['keypoints'][12][1]
        #bi_shoudler = eucliDist(pre_rishoulder,pre_leshoulder)
        #if this_frame_velocity[11]>bi_shoudler/2 and this_frame_velocity[12]>bi_shoudler/2: #表示上半身可能跳了
        #if this_frame_velocity[11]>=0.1 and this_frame_velocity[12]>=0.1:
        #if dis_x[13]>0.25 and dis_x[14]>0.25:    #shoulder應該可以用x來判斷有沒有跳，但是hip要想一下
        #if dis_x[11]>abs(bi_x_dis)-0.001 and dis_x[12]>abs(bi_x_dis)-0.001:
        #     print(f"上半在{frame}跳了")
        #     save_frame_up = frame
        #     jump_up.append(frame)
            
        #     for j in range(0,23):
        #         speed=[]
        #         for k in range(1,11): #打算用前五幀的速度來預測
        #             getv = np.array(change_data[-k]['keypoints'][j][1:4])-np.array(change_data[-k-2]['keypoints'][j][1:4])
        #             speed.append(getv)
        #         mean_v = np.mean(speed, axis=0)
        #         #print("v",mean_v)
        #         #print(speed)
        #         cha = np.array(change_data[-2]['keypoints'][j][1:4]) + mean_v
        #         #這一幀 = 原本的這一幀+(上一幀-上上一幀)，看起來好像不行
        #         #print(j, cha)
        #         new_keypoint = [j] + cha.tolist() + [data[frame]['keypoints'][j][4]]
        #         new_keypoints.append(new_keypoint)
            
        # else:
        #     new_keypoints.extend(data[frame]['keypoints'][0:23])

        # #if this_frame_velocity[25]>=0.08 and this_frame_velocity[26]>=0.08: #表示下半身可能跳了
        # if dis_x[25]>=0.09 and dis_x[26]>=0.09:
        #     print(f"下半在{frame}跳了")
        #     jump_down.append(frame)
            
        #     for j in range(23,33):
                
        #         speed=[]
        #         for k in range(1,11): #打算用前五幀的速度來預測
        #             getv = np.array(change_data[-k]['keypoints'][j][1:4])-np.array(change_data[-k-2]['keypoints'][j][1:4])
        #             speed.append(getv)
        #         mean_v = np.mean(speed, axis=0)
        #         #print("v",mean_v)
        #         #print(speed)
        #         cha = np.array(change_data[-2]['keypoints'][j][1:4]) + mean_v
        #         #cha = np.array(change_data[-1]['keypoints'][j][1:4]) + (np.array(change_data[-1]['keypoints'][j][1:4])-np.array(change_data[-2]['keypoints'][j][1:4]))
        #         #這一幀 = 原本的這一幀+(上一幀-上上一幀)
        #         new_keypoint = [j] + cha.tolist() + [data[frame]['keypoints'][j][4]]
        #         new_keypoints.append(new_keypoint)
                
        # else:
        #     new_keypoints.extend(data[frame]['keypoints'][23:])
        # temp_dic['keypoints'] = new_keypoints
        # #print("temp",temp_dic)
        # change_data.append(temp_dic)
        #save_vel.append(this_frame_velocity)

        #threshold = 0.294 #freestyle是0.294，acting是0.288
        threshold = get_thres #用前面算速度去抓，抓第一個大於0.25的當threshold
        new_keypoints = []
        #先看15和16的可見度有沒有超過0.85，有的話才看，沒有的話就只看有超過0.85的，這樣才比較準
        if data[frame]['keypoints'][15][4]>=0.85 and data[frame]['keypoints'][16][4]>=0.85: 
            if this_frame_velocity[15] >= threshold and this_frame_velocity[16] >= threshold: #如果該幀移動的速度超過閾值，表示有跳
                #直接全身換，不要分上下半身
            #if this_frame_velocity[15]>=0.265 and this_frame_velocity[16]>=0.265:
            #if (this_frame_velocity[23]>=0.1 and this_frame_velocity[24]>=0.1) : #直接全身換，不要分上下半身
            #if dis_x[23]>=0.1 or dis_x[24]>=0.1:
                #print(f"在{frame}跳了")
                save_frame_up = frame
                jump_up.append(frame)

                """ 交換用
                change_data[frame]['keypoints'][0][1:4] = data[frame]['keypoints'][0][1:4]
                change_data[frame]['keypoints'][1][1:4] = data[frame]['keypoints'][4][1:4]
                change_data[frame]['keypoints'][2][1:4] = data[frame]['keypoints'][5][1:4]
                change_data[frame]['keypoints'][3][1:4] = data[frame]['keypoints'][6][1:4]
                change_data[frame]['keypoints'][4][1:4] = data[frame]['keypoints'][1][1:4]
                change_data[frame]['keypoints'][5][1:4] = data[frame]['keypoints'][2][1:4]
                change_data[frame]['keypoints'][6][1:4] = data[frame]['keypoints'][3][1:4]
                change_data[frame]['keypoints'][7][1:4] = data[frame]['keypoints'][8][1:4]
                change_data[frame]['keypoints'][8][1:4] = data[frame]['keypoints'][7][1:4]
                change_data[frame]['keypoints'][9][1:4] = data[frame]['keypoints'][10][1:4]
                change_data[frame]['keypoints'][10][1:4] = data[frame]['keypoints'][9][1:4]
                change_data[frame]['keypoints'][11][1:4] = data[frame]['keypoints'][12][1:4]
                change_data[frame]['keypoints'][12][1:4] = data[frame]['keypoints'][11][1:4]
                change_data[frame]['keypoints'][13][1:4] = data[frame]['keypoints'][14][1:4]
                change_data[frame]['keypoints'][14][1:4] = data[frame]['keypoints'][13][1:4]
                change_data[frame]['keypoints'][15][1:4] = data[frame]['keypoints'][16][1:4]
                change_data[frame]['keypoints'][16][1:4] = data[frame]['keypoints'][15][1:4]
                change_data[frame]['keypoints'][17][1:4] = data[frame]['keypoints'][18][1:4]
                change_data[frame]['keypoints'][18][1:4] = data[frame]['keypoints'][17][1:4]
                change_data[frame]['keypoints'][19][1:4] = data[frame]['keypoints'][20][1:4]
                change_data[frame]['keypoints'][20][1:4] = data[frame]['keypoints'][19][1:4]
                change_data[frame]['keypoints'][21][1:4] = data[frame]['keypoints'][22][1:4]
                change_data[frame]['keypoints'][22][1:4] = data[frame]['keypoints'][21][1:4]
                change_data[frame]['keypoints'][23][1:4] = data[frame]['keypoints'][24][1:4]
                change_data[frame]['keypoints'][24][1:4] = data[frame]['keypoints'][23][1:4]
                change_data[frame]['keypoints'][25][1:4] = data[frame]['keypoints'][26][1:4]
                change_data[frame]['keypoints'][26][1:4] = data[frame]['keypoints'][25][1:4]
                change_data[frame]['keypoints'][27][1:4] = data[frame]['keypoints'][28][1:4]
                change_data[frame]['keypoints'][28][1:4] = data[frame]['keypoints'][27][1:4]
                change_data[frame]['keypoints'][29][1:4] = data[frame]['keypoints'][30][1:4]
                change_data[frame]['keypoints'][30][1:4] = data[frame]['keypoints'][29][1:4]
                change_data[frame]['keypoints'][31][1:4] = data[frame]['keypoints'][32][1:4]
                change_data[frame]['keypoints'][32][1:4] = data[frame]['keypoints'][31][1:4]
                """

                new_keypoints.append(data[frame]['keypoints'][0]) #0
                new_keypoints.append(data[frame]['keypoints'][4]) #1
                new_keypoints.append(data[frame]['keypoints'][5]) #2
                new_keypoints.append(data[frame]['keypoints'][6]) #3
                new_keypoints.append(data[frame]['keypoints'][1]) #4
                new_keypoints.append(data[frame]['keypoints'][2]) #5
                new_keypoints.append(data[frame]['keypoints'][3]) #6
                new_keypoints.append(data[frame]['keypoints'][8]) #7
                new_keypoints.append(data[frame]['keypoints'][7]) #8
                new_keypoints.append(data[frame]['keypoints'][10]) #9
                new_keypoints.append(data[frame]['keypoints'][9]) #10
                new_keypoints.append(data[frame]['keypoints'][12]) #11
                new_keypoints.append(data[frame]['keypoints'][11]) #12
                new_keypoints.append(data[frame]['keypoints'][14]) #13
                new_keypoints.append(data[frame]['keypoints'][13]) #14
                new_keypoints.append(data[frame]['keypoints'][16]) #15
                new_keypoints.append(data[frame]['keypoints'][15]) #16
                new_keypoints.append(data[frame]['keypoints'][18]) #17
                new_keypoints.append(data[frame]['keypoints'][17]) #18
                new_keypoints.append(data[frame]['keypoints'][20]) #19
                new_keypoints.append(data[frame]['keypoints'][19]) #20
                new_keypoints.append(data[frame]['keypoints'][22]) #21
                new_keypoints.append(data[frame]['keypoints'][21]) #22
                new_keypoints.append(data[frame]['keypoints'][24]) #23
                new_keypoints.append(data[frame]['keypoints'][23]) #24
                new_keypoints.append(data[frame]['keypoints'][26]) #25
                new_keypoints.append(data[frame]['keypoints'][25]) #26
                new_keypoints.append(data[frame]['keypoints'][28]) #27
                new_keypoints.append(data[frame]['keypoints'][27]) #28
                new_keypoints.append(data[frame]['keypoints'][30]) #29
                new_keypoints.append(data[frame]['keypoints'][29]) #30
                new_keypoints.append(data[frame]['keypoints'][32]) #31
                new_keypoints.append(data[frame]['keypoints'][31]) #32



            else:
                for i in range(33):
                    new_keypoints.append(data[frame]['keypoints'][i])

        elif data[frame]['keypoints'][15][4]<0.85 and data[frame]['keypoints'][16][4]>=0.85: 
            #手可見度低，不能用手判
            if this_frame_velocity[16]>=threshold:
                print(f"在{frame}跳了")
                save_frame_up = frame
                jump_up.append(frame)
                new_keypoints.append(data[frame]['keypoints'][0]) #0
                new_keypoints.append(data[frame]['keypoints'][4]) #1
                new_keypoints.append(data[frame]['keypoints'][5]) #2
                new_keypoints.append(data[frame]['keypoints'][6]) #3
                new_keypoints.append(data[frame]['keypoints'][1]) #4
                new_keypoints.append(data[frame]['keypoints'][2]) #5
                new_keypoints.append(data[frame]['keypoints'][3]) #6
                new_keypoints.append(data[frame]['keypoints'][8]) #7
                new_keypoints.append(data[frame]['keypoints'][7]) #8
                new_keypoints.append(data[frame]['keypoints'][10]) #9
                new_keypoints.append(data[frame]['keypoints'][9]) #10
                new_keypoints.append(data[frame]['keypoints'][12]) #11
                new_keypoints.append(data[frame]['keypoints'][11]) #12
                new_keypoints.append(data[frame]['keypoints'][14]) #13
                new_keypoints.append(data[frame]['keypoints'][13]) #14
                new_keypoints.append(data[frame]['keypoints'][16]) #15
                new_keypoints.append(data[frame]['keypoints'][15]) #16
                new_keypoints.append(data[frame]['keypoints'][18]) #17
                new_keypoints.append(data[frame]['keypoints'][17]) #18
                new_keypoints.append(data[frame]['keypoints'][20]) #19
                new_keypoints.append(data[frame]['keypoints'][19]) #20
                new_keypoints.append(data[frame]['keypoints'][22]) #21
                new_keypoints.append(data[frame]['keypoints'][21]) #22
                new_keypoints.append(data[frame]['keypoints'][24]) #23
                new_keypoints.append(data[frame]['keypoints'][23]) #24
                new_keypoints.append(data[frame]['keypoints'][26]) #25
                new_keypoints.append(data[frame]['keypoints'][25]) #26
                new_keypoints.append(data[frame]['keypoints'][28]) #27
                new_keypoints.append(data[frame]['keypoints'][27]) #28
                new_keypoints.append(data[frame]['keypoints'][30]) #29
                new_keypoints.append(data[frame]['keypoints'][29]) #30
                new_keypoints.append(data[frame]['keypoints'][32]) #31
                new_keypoints.append(data[frame]['keypoints'][31]) #32

            else:
                for i in range(33):
                    new_keypoints.append(data[frame]['keypoints'][i])

        elif data[frame]['keypoints'][16][4]<0.85 and data[frame]['keypoints'][15][4]>=0.85:
            if this_frame_velocity[15]>=threshold:
                print(f"在{frame}跳了")
                save_frame_up = frame
                jump_up.append(frame)
                new_keypoints.append(data[frame]['keypoints'][0]) #0
                new_keypoints.append(data[frame]['keypoints'][4]) #1
                new_keypoints.append(data[frame]['keypoints'][5]) #2
                new_keypoints.append(data[frame]['keypoints'][6]) #3
                new_keypoints.append(data[frame]['keypoints'][1]) #4
                new_keypoints.append(data[frame]['keypoints'][2]) #5
                new_keypoints.append(data[frame]['keypoints'][3]) #6
                new_keypoints.append(data[frame]['keypoints'][8]) #7
                new_keypoints.append(data[frame]['keypoints'][7]) #8
                new_keypoints.append(data[frame]['keypoints'][10]) #9
                new_keypoints.append(data[frame]['keypoints'][9]) #10
                new_keypoints.append(data[frame]['keypoints'][12]) #11
                new_keypoints.append(data[frame]['keypoints'][11]) #12
                new_keypoints.append(data[frame]['keypoints'][14]) #13
                new_keypoints.append(data[frame]['keypoints'][13]) #14
                new_keypoints.append(data[frame]['keypoints'][16]) #15
                new_keypoints.append(data[frame]['keypoints'][15]) #16
                new_keypoints.append(data[frame]['keypoints'][18]) #17
                new_keypoints.append(data[frame]['keypoints'][17]) #18
                new_keypoints.append(data[frame]['keypoints'][20]) #19
                new_keypoints.append(data[frame]['keypoints'][19]) #20
                new_keypoints.append(data[frame]['keypoints'][22]) #21
                new_keypoints.append(data[frame]['keypoints'][21]) #22
                new_keypoints.append(data[frame]['keypoints'][24]) #23
                new_keypoints.append(data[frame]['keypoints'][23]) #24
                new_keypoints.append(data[frame]['keypoints'][26]) #25
                new_keypoints.append(data[frame]['keypoints'][25]) #26
                new_keypoints.append(data[frame]['keypoints'][28]) #27
                new_keypoints.append(data[frame]['keypoints'][27]) #28
                new_keypoints.append(data[frame]['keypoints'][30]) #29
                new_keypoints.append(data[frame]['keypoints'][29]) #30
                new_keypoints.append(data[frame]['keypoints'][32]) #31
                new_keypoints.append(data[frame]['keypoints'][31]) #32

            else:
                for i in range(33):
                    new_keypoints.append(data[frame]['keypoints'][i])
        else:
            for i in range(33):
                new_keypoints.append(data[frame]['keypoints'][i])
        #     if frame-1 in jump_up: #延續前一幀，前一幀錯就認定錯，對就對
        #         jump_up.append(frame)
        #         new_keypoints.append(data[frame]['keypoints'][0]) #0
        #         new_keypoints.append(data[frame]['keypoints'][4]) #1
        #         new_keypoints.append(data[frame]['keypoints'][5]) #2
        #         new_keypoints.append(data[frame]['keypoints'][6]) #3
        #         new_keypoints.append(data[frame]['keypoints'][1]) #4
        #         new_keypoints.append(data[frame]['keypoints'][2]) #5
        #         new_keypoints.append(data[frame]['keypoints'][3]) #6
        #         new_keypoints.append(data[frame]['keypoints'][8]) #7
        #         new_keypoints.append(data[frame]['keypoints'][7]) #8
        #         new_keypoints.append(data[frame]['keypoints'][10]) #9
        #         new_keypoints.append(data[frame]['keypoints'][9]) #10
        #         new_keypoints.append(data[frame]['keypoints'][12]) #11
        #         new_keypoints.append(data[frame]['keypoints'][11]) #12
        #         new_keypoints.append(data[frame]['keypoints'][14]) #13
        #         new_keypoints.append(data[frame]['keypoints'][13]) #14
        #         new_keypoints.append(data[frame]['keypoints'][16]) #15
        #         new_keypoints.append(data[frame]['keypoints'][15]) #16
        #         new_keypoints.append(data[frame]['keypoints'][18]) #17
        #         new_keypoints.append(data[frame]['keypoints'][17]) #18
        #         new_keypoints.append(data[frame]['keypoints'][20]) #19
        #         new_keypoints.append(data[frame]['keypoints'][19]) #20
        #         new_keypoints.append(data[frame]['keypoints'][22]) #21
        #         new_keypoints.append(data[frame]['keypoints'][21]) #22
        #         new_keypoints.append(data[frame]['keypoints'][24]) #23
        #         new_keypoints.append(data[frame]['keypoints'][23]) #24
        #         new_keypoints.append(data[frame]['keypoints'][26]) #25
        #         new_keypoints.append(data[frame]['keypoints'][25]) #26
        #         new_keypoints.append(data[frame]['keypoints'][28]) #27
        #         new_keypoints.append(data[frame]['keypoints'][27]) #28
        #         new_keypoints.append(data[frame]['keypoints'][30]) #29
        #         new_keypoints.append(data[frame]['keypoints'][29]) #30
        #         new_keypoints.append(data[frame]['keypoints'][32]) #31
        #         new_keypoints.append(data[frame]['keypoints'][31]) #32
        #     else:
        #         for i in range(33):
        #             new_keypoints.append(data[frame]['keypoints'][i])


        temp_dic['keypoints'] = new_keypoints
        #print(temp_dic)
        change_data.append(temp_dic)

    jump_frame = [x+2900 for x in jump_up] #確認到底是哪幾幀跳了，這邊acting的時候是從2900開始地所以+2900，freestyle那邊是+1900
    #jump_frame = [x+1900 for x in jump_up]
    print(jump_frame)

    
    




    
    #print(jump_down)

    return change_data


        
        
    
#處理MEDIAPIPE, 判斷有沒有左右亂跳，這邊有要先算相對速度
def process_vel(jsonfile):
    data = readjson(jsonfile)
    all_v = []
    for frame in range(len(data)): 
        print("frame", frame)
        this_frame_velocity = [] #存這幀的每個關節的相對速度
        for i in range(33):#這邊是我拿來算什麼都沒處理時的相對速度，這樣我才可以找閾值

            pre_position = data[frame-1]['keypoints'][i][1:4]
            #print(pre_position)
            now_position =data[frame]['keypoints'][i][1:4]
            #print(now_position)
            v= eucliDist(pre_position, now_position)
                    
            #print(v)
            #print(type(v))
            this_frame_velocity.append(v)
        all_v.append(this_frame_velocity) #這幀的所有相對速度都放進all_v
    temp_json = jsonfile.split('.')[0]+'_vel.json'  
    write_json(all_v,temp_json)
    csv_path = jsonfile.split('.')[0]+'_vel.csv'
   
    print("saving:",csv_path)
    header = ['frame']
    get_threshold = 0
    for i in range(33):
        header.append(str(i))  #標頭，frame, 0 ,1, 2, .....的關節編碼
    with open(csv_path, 'w', newline='') as csvfile:  #存速度的東西
        writer = csv.writer(csvfile)
        writer.writerow(header) #先印標題
        for i in range(len(all_v)): #所有幀的速度
            towrite = [i+2900] #acting那邊因為是從第2900幀開始，所以這邊+2900，freestyle的話就是+1900，因為這邊是我要寫現在是第幾幀
            #towrite = [i+1900]
            for j in range(33): #一個一個關節存下來
                towrite.append(all_v[i][j])
            writer.writerow(towrite)
            if get_threshold == 0 and (all_v[i][15]>0.25 and all_v[i][16]>0.25) and i!=0: #如果都還沒抓，那就抓第一個大於0.25的當閾值(第一幀不能算)
                if all_v[i][15]>all_v[i][16]:
                    get_threshold = all_v[i][16]
                else:
                    get_threshold = all_v[i][15]
    
    

    newdata = process_change(data,get_threshold) #處理亂跳的部分
    change_file = jsonfile.split('.')[0]+'_ch.json'#儲存改好跳的json
    write_json(newdata,change_file)
    print("抓",get_threshold)





   

if __name__ == "__main__":
    #mediajson = "E://things/master/pose3d/result/TC_S1_freestyle1_cam1/freestyle1900.json"
    mediajson = "E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900.json"
    process_vel(mediajson)
    #get_direction(mediajson)
    
    #here = readjson("E://things/master/pose3d/result/TC_S1_acting1_cam1/TC_S1_acting1_cam1_vel.json")
    #print(here)
