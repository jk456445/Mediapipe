import numpy as np
import json
import csv

def write_json(write, file):
    output_file = file
    write_list = write
     # 寫入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')



def eucliDist(A, B):
    
    first = np.array(A)#只抓X,Y,Z軸
    #print(first)
    second = np.array(B)
    dist = np.sqrt(np.sum((first-second)**2))
    #print("dist:", dist)
    return dist


def calculate_ratio(list1, list2, x): #拿來算現在的值對過去是多少
    list1 = [0,1]
    enter_ = x-list2[0]
    if enter_<=0:
        enter_=0
    ratio = enter_/list2[1]-list2[0] #比例，可以直接返回是因為1-0也是1，所以這個比例對過去另一邊就是值了

    return ratio



def exponential_moving_average(prices, period, weighting_factor=0.2):
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period]) 
    ema[0:period-1] = prices[0:period-1] #我給他改成前面維持原樣，不燃就會是0
    ema[period - 1] = sma #因為有10個才會開始算，所以前面的都不會抓
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return ema




def exponential_moving_average_adaptive(prices, period):
    prices = np.array(prices)
    
    ema = np.zeros((len(prices),3))
    sma = np.average(prices[:period], axis=0) #我改這樣才會是個別平均x,y,z
    #print(sma)
    ema[0:period-1] = prices[0:period-1] #我給他改成前面維持原樣，不然就會是0
    
    ema[period - 1] = sma #因為有10個才會開始算，所以前面的都不會抓
    #print(ema[period - 1])
    #print(ema[0:period])
    #varia = np.var(prices[:period], axis=0)
    #print(varia)
    weighting_factor = 0.2  #老師說要看前面的std來看他到底是抖動還是真的有變化，然後用這個來決定weighting factor
    all_variace = []
    for i in range(period, len(prices)):
        #print("i",i)
        #看10個點跟平均點的距離的variance
        sumx = np.sum(prices[i-period:i][0])/period
        sumy = np.sum(prices[i-period:i][1])/period
        sumz = np.sum(prices[i-period:i][2])/period
        mean_point = np.array([sumx,sumy,sumz]) #10個點的平均點
        point10dis = []
        for j in range(i-period,i):
            this_po = eucliDist(mean_point,prices[j])
            point10dis.append(this_po) #10個點跟平均點之間的距離，暫時放到list，下面要拿來算variance

        #print(len(point10dis))
        #print(point10dis)
        #print(np.var(point10dis))
        now10var = np.var(point10dis)
        all_variace.append(now10var)
         #目前測出投球那邊最大variace 0.03，最小1e-9f
        vari_list = [1e-9, 0.05]
        ratio = calculate_ratio([0,1],vari_list,now10var)
        #print(f"{ratio}")
        weighting_factor = ratio
        if now10var<=1e-6:
            weighting_factor = 0.3
        elif 1e-3>=now10var>1e-6:
            weighting_factor = 0.5
        else:
            weighting_factor = 0.7


        # varia = np.var(prices[i-period:i], axis=0)
        # varia2 = np.sum(varia)  #加起來，看var
        # if varia2 >= 0.001:#大概有移動?想捕捉短期波動要用較小的alpha值，長期趨勢用較大的alpha值
        #     weighting_factor = 0.3
        # else:
        #     weighting_factor = 0.7
        
        
        #print("v",varia)
        # print("v3",varia2)
        
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    #print(f"max variance:{max(all_variace)}, min variance:{min(all_variace)}")

    return ema




def readjson(path):
    with open(path, newline='') as jsonfile:
        data = json.load(jsonfile)
        # 或者這樣
        # data = json.loads(jsonfile.read())
        #print(data)
        return data



def one_ema(json_file):
    

        #下面是讀取數據跟存數據的部分
    data = readjson(json_file)
    # with open(json_file, newline='') as jsonfile:
    #     data = json.load(jsonfile)
        # 或者這樣
        # data = json.loads(jsonfile.read())
        #print(data)
    temp_all = []
    all_visibility = []
    for key in range(33):
        print("joint:",key)
        temp_thiskey_x = []
        temp_thiskey_y = []
        temp_thiskey_z = []
        visib = []
        this_joint =[]
        for i in range(len(data)):
            #print("frame:",i)                
            #temp_store.append(data[i]['keypoints'][11][1]) #左肩試試看
            pos = []
            # pos.append(data[i]['keypoints'][key][1]*100) #暫時放這個關節的x
            # pos.append(data[i]['keypoints'][key][2]*100)  #發現有沒有換成公分都沒差
            # pos.append(data[i]['keypoints'][key][3]*100)
            pos.append(data[i]['keypoints'][key][1]) #暫時放這個關節的x
            pos.append(data[i]['keypoints'][key][2])
            pos.append(data[i]['keypoints'][key][3])
            visib.append(data[i]['keypoints'][key][4])
            
            this_joint.append(pos)
        returnx = exponential_moving_average_adaptive(this_joint,10)

        #print(len(returnx))
        for j in range(len(returnx)):  #把預測出來的x,y,z放在一起
            #print(j)
            #print(returnx[j])
            #here=[key,returnx[j][0]/100,returnx[j][1]/100,returnx[j][2]/100] #關節點，x,y,z
            here=[key,returnx[j][0],returnx[j][1],returnx[j][2],visib[j]] #關節點，x,y,z
            #print(here)
            temp_all.append(here)
        #print(f"visibility:{visib}")
        
        all_visibility.append(visib)
    csv_path = json_file.split('.')[0]+'visib.csv'
   
    print("saving:",csv_path)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['frame','11','12','13','14', '15','16', '23','24','25', '26','27', '28']
        writer.writerow(header)
        for i in range(len(all_visibility[0])): 
            writer.writerow([i,all_visibility[11][i],all_visibility[12][i],all_visibility[13][i],all_visibility[14][i],
                            all_visibility[15][i],all_visibility[16][i],all_visibility[23][i],all_visibility[24][i],all_visibility[25][i],
                            all_visibility[26][i],all_visibility[27][i],all_visibility[28][i]])
                
            

    dict_all=[] #放所有dict的地方，想把它弄成跟之前的格式一樣
    
    for frame in range(len(data)):
        #print("frame")
        #print(frame)
        temp_dic = {'frame':frame+1, 'keypoints':[]}
        # temp_dic['frame'] = frame+1
        # temp_dic['keypoints']=[]
        for lll in range(len(temp_all)):
            if frame == lll%len(data):
                #print(temp_all[lll])
                temp_dic['keypoints'].append(temp_all[lll])
        #print("tempdic")
        #print(temp_dic)
        dict_all.append(temp_dic)
        #print("predict all")
        #print(dict_all)

    #output_json = json_file.split('.')[0]+'_UKF.json'
    #output_json = json_file.split('.')[0]+'_UKF1.json'
    #output_json = json_file.split('.')[0]+'_UKF2.json'
    output_json = json_file.split('.')[0]+'EMA2.json'
    #output_json = json_file.split('.')[0]+'_UKFA_ESPCN.json'
        #print("output:",output_json)
    write_json(dict_all,output_json)
        #粒子濾波的版本沒有放visibility
    #print(dict_all)
    #print("all visibility:",all_visibility)
    return output_json



if __name__ == "__main__":
    #prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
    #period = 5
    #weighting_factor = 0.2
    #ema = exponential_moving_average(prices, period, weighting_factor)
    #print(ema)
    #json_file1 = "E://things/master/pose3d/result/TC_S1_freestyle1_cam1/freestyle1900.json"
    #json_file1 = "E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900.json"
    json_file1 = "E://things/master/pose3d/result/S001C001P001R001A003_rgb/S001C001P001R001A003_rgb_sharpen.json"

    returnjson = one_ema(json_file1) 

    
    