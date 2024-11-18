import numpy as np
import json

def write_json(write, file):
    output_file = file
    write_list = write
     # 寫入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')




def unscented_kalman_filter(x0, P0, Q, R, measurements, process_model, measurement_model):
    n = len(x0)
    m = len(measurements)
    
    x = x0
    P = P0
    
    # 保存每个时间点的预测结果和更新后的状态
    predicted_states = []
    updated_states = []
    
    for i in range(m):
        # 预测步骤
        
        P_prev = P
        x_pred, P_pred = ukf_predict(x, P, Q, process_model)
        predicted_states.append(x_pred.copy())  # 保存预测结果
        
        # 更新步骤
        z = measurements[i][0:3]
        vis = measurements[i][3]
        #z = measurements[i]
        #x, P = ukf_update(x_pred, P_pred, z, R, measurement_model)
        x, P ,Q, R = ukf_update(x_pred, P_pred, z, R, measurement_model,P_prev,Q,vis)
       
        updated_states.append(x.copy())  # 保存更新后的状态

        
    return predicted_states, updated_states

def ukf_predict(x, P, Q, process_model):
    # UKF 参数设置
    alpha = 0.1
    beta = 2
    kappa = 0
    
    n = len(x)
    L = n * 2 + 1
    
    # Sigma 点生成
    sigma_points = np.zeros((n, L))
    sigma_points[:, 0] = x
    sqrt_P = np.linalg.cholesky(P * (n + kappa))
    for i in range(n):
        sigma_points[:, i+1] = x + sqrt_P[:, i]
        sigma_points[:, i+1+n] = x - sqrt_P[:, i]
    


    # 预测 Sigma 点
    predicted_sigma_points = np.zeros((n, L))
    for i in range(L):
        predicted_sigma_points[:, i] = process_model(sigma_points[:, i])
    
    # 预测均值和协方差
    x_pred = np.sum(predicted_sigma_points, axis=1) / L
    P_pred = np.zeros((n, n))
    for i in range(L):
        diff = predicted_sigma_points[:, i] - x_pred
        P_pred += np.outer(diff, diff) / L
    P_pred += Q
    
    return x_pred, P_pred

#def ukf_update(x, P, z, R, measurement_model):
def ukf_update(x, P, z, R, measurement_model,prev,Q,visibility): #多加一個自適應調整P,Q,R
    # UKF 参数设置
    alpha = 1.2
    beta = 2
    kappa = 2
    
    n = len(x)
    #print(n)
    L = n * 2 + 1
    
    # Sigma 点生成
    sigma_points = np.zeros((n, L))
    sigma_points[:, 0] = x
    sqrt_P = np.linalg.cholesky(P * (n + kappa))
    for i in range(n):
        sigma_points[:, i+1] = x + sqrt_P[:, i]
        sigma_points[:, i+1+n] = x - sqrt_P[:, i]
    
    # 预测测量 Sigma 点
    predicted_measurement_sigma_points = np.zeros((len(z), L))
    for i in range(L):
        predicted_measurement_sigma_points[:, i] = measurement_model(sigma_points[:, i])
    
    # 预测测量均值和协方差
    z_pred = np.sum(predicted_measurement_sigma_points, axis=1) / L
    Pzz = np.zeros((len(z), len(z)))
    Pxz = np.zeros((n, len(z)))
    for i in range(L):
        diff_x = sigma_points[:, i] - x
        diff_z = predicted_measurement_sigma_points[:, i] - z_pred
        Pzz += np.outer(diff_z, diff_z) / L
        Pxz += np.outer(diff_x, diff_z) / L
    
    # Kalman 增益
    K = np.dot(Pxz, np.linalg.inv(Pzz + R))
    
    # 更新状态和协方差
    x += np.dot(K, (z - z_pred))
    P -= np.dot(K, np.dot(Pzz, K.T))

    # 计算变化比率，這邊以下是自適應的版本
    #change_ratio_Q = np.maximum(1, np.abs(np.diag(P-prev)) / np.diag(Q))
    min_visibility = 0.1  # 可见度的最小值
    max_visibility = 1.0  # 可见度的最大值
    min_noise_strength = 0.01  # 测量噪声的最小强度
    max_noise_strength = 1  # 测量噪声的最大强度
    #change_ratio_R = np.maximum(1, np.abs(z - z_pred) / np.diag(R))
    # 根据可见度线性地调整测量噪声强度
    # noise_strength_R = np.interp(visibility, [min_visibility, max_visibility], [max_noise_strength, min_noise_strength])
    # noise_strength_Q = np.interp(visibility, [min_visibility, max_visibility], [min_noise_strength, max_noise_strength])
    # # 将调整后的测量噪声强度应用于测量噪声协方差矩阵的对角元素
    # R_adjusted = np.diag(np.full(R.shape[0], noise_strength_R))
    # R = R_adjusted
    # Q =  np.diag(np.full(Q.shape[0], noise_strength_Q))

    # if visibility<=0.7:
    #     R = np.eye(3) * 0.0005
    # else:
    #     R = np.eye(3) *10

    
    # 更新 Q 和 R
    #Q = Q * change_ratio_Q
    #R = R * change_ratio_R

    
    return x, P,Q,R
    #return x, P

# 示例：模拟系统状态转移和测量模型
def process_model(x):
    # 三维状态转移模型，这里简单地假设状态变化量等于 x 的值
    return x

def measurement_model(x):
    # 三维测量模型，这里简单地假设测量值等于 x 的值
    return x


def UKFmain(measure):
        # 初始状态和协方差
    x0 = np.array([0, 0, 0])
    #P0 = np.eye(3)*0.01
    P0 = np.eye(3)
    #P0 = np.eye(3)*0.001

    # 过程噪声和测量噪声协方差
    Q = np.eye(3) * 0.1
    #Q = np.eye(3) * 1
    #Q = np.eye(3) * 0.1
    #R = np.eye(3) * 0.1 
    R = np.eye(3) * 10
    # 测量值
    #measurements = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    measurements = measure

    def calculate_observation_covariance(measurements):
        """
        计算观测噪音协方差矩阵R
        参数:
            measurements (numpy array): 观测数据, shape = (num_observations, num_variables)
        返回:
            R (numpy array): 观测噪音协方差矩阵, shape = (num_variables, num_variables)
        """
        # 计算均值
        #print(len(measurements))
        measurements = [measurements[i][0:3] for i in range(len(measurements))]
        mean_measurements = np.mean(measurements, axis=0)
        #print(mean_measurements)

        # 计算误差
        errors = measurements - mean_measurements

        # 计算协方差矩阵
        R = np.cov(errors, rowvar=False)

        return R
    
    R_ = calculate_observation_covariance(measurements)
    #print("R_",R_)
    x0 = np.array(measurements[0][0:3])
    #x0 = np.array(measurements[0])
    
    # 运行无迹卡尔曼滤波
    predicted_states, updated_states = unscented_kalman_filter(x0, P0, Q, R, measurements, process_model, measurement_model)
    
    #for i in range(len(predicted_states)):
        #print(f"Time step {i + 1}: Predicted state = {predicted_states[i]}, Updated state = {updated_states[i]}")
        #update state才識更新狀態
    return updated_states

def GPY(json_file):
        #下面是讀取數據跟存數據的部分
    with open(json_file, newline='') as jsonfile:
        data = json.load(jsonfile)
        # 或者這樣
        # data = json.loads(jsonfile.read())
        #print(data)
        temp_all = []
        for key in range(33):
            #print(key)
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
                pos.append(data[i]['keypoints'][key][4])
                this_joint.append(pos)
            returnx = UKFmain(this_joint)
            #print(len(returnx))
            for j in range(len(returnx)):  #把預測出來的x,y,z放在一起
                #print(j)
                #print(returnx[j])
                #here=[key,returnx[j][0]/100,returnx[j][1]/100,returnx[j][2]/100] #關節點，x,y,z
                here=[key,returnx[j][0],returnx[j][1],returnx[j][2],this_joint[j][3]] #關節點，x,y,z
                #print(here)
                temp_all.append(here)

        dict_all=[] #放所有dict的地方，想把它弄成跟之前的格式一樣
        
        for frame in range(len(data)):
            print("frame")
            print(frame)
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
        output_json = json_file.split('.')[0]+'_UKF_pic.json'
        #output_json = json_file.split('.')[0]+'_UKFA_ESPCN.json'
            #print("output:",output_json)
        write_json(dict_all,output_json)
            #粒子濾波的版本沒有放visibility
        #print(dict_all)

# 主程序
if __name__ == "__main__":

    json_file = "E://things/master/pose3d/result/S001C001P001R001A003_rgb/S001C001P001R001A003_rgb_sharpenEMA.json"
    #json_file = "E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900_chEMA.json" #計算用
    #json_file = "E://things/master/pose3d/result/TC_S1_freestyle1_cam1/freestyle1900EMA2.json"
    #json_file = "E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900_chEMA.json"
    #json_file = "E://things/master/pose3d/video/media/faa/myData1201/myData1201EMA.json"


    #json_file = input("json:") #呼叫用
    GPY(json_file)

