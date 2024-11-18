import numpy as np
import json

def write_json(write, file):
    output_file = file
    write_list = write
    # 写入
    with open(output_file, 'w') as json_file:
        json.dump(write_list, json_file)
    
    print(f'{output_file}寫入完成')

def unscented_kalman_filter(x0, P0, Q, R, measurements, process_model, measurement_model, dt):
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
        x_pred, P_pred = ukf_predict(x, P, Q, process_model, dt, measurements, i)
        predicted_states.append(x_pred.copy())  # 保存预测结果
        
        # 更新步骤
        z = measurements[i][0:3]
        vis = measurements[i][3]
        x, P, Q, R = ukf_update(x_pred, P_pred, z, R, measurement_model, P_prev, Q, vis)
       
        updated_states.append(x.copy())  # 保存更新后的状态

    return predicted_states, updated_states

def ukf_predict(x, P, Q, process_model, dt, measurements, i):
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
        predicted_sigma_points[:, i] = process_model(sigma_points[:, i], dt, measurements, i)
    
    # 预测均值和协方差
    x_pred = np.sum(predicted_sigma_points, axis=1) / L
    P_pred = np.zeros((n, n))
    for i in range(L):
        diff = predicted_sigma_points[:, i] - x_pred
        P_pred += np.outer(diff, diff) / L
    P_pred += Q
    
    return x_pred, P_pred

def ukf_update(x, P, z, R, measurement_model, prev, Q, visibility):
    # UKF 参数设置
    #alpha = 0.1
    alpha = 1.2
    beta = 2
    #kappa = 0
    kappa = 2
    
    n = len(x)
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

    # 自适应调整
    min_visibility = 0.1
    max_visibility = 1.0
    min_noise_strength = 0.01
    max_noise_strength = 1
    return x, P, Q, R

# 状态转移函数
def process_model(x, dt, measurements, i):

    if i == 0:
        velocity = np.array([0, 0, 0])  # 初始速度假设
    else:
        current_measurement = np.array(measurements[i][0:3])
        #previous_measurement = np.array(measurements[i-1][0:3])
        previous_measurement = x[:3]  # 上一帧的预测位置
        velocity = (current_measurement - previous_measurement) / dt
    # 预测位置和速度的更新
    next_position = x[:3] + velocity * dt  # 更新位置
    next_state = np.concatenate((next_position, velocity))  # 将位置和速度组合成新的状态向量
    #return x + velocity * dt
    return next_state

# 测量模型
def measurement_model(x):
    return x[:3]

def UKFmain(measure):
    # 初始状态和协方差
   # x0 = np.array([0, 0, 0])
    x0 = np.zeros(6)
    #P0 = np.eye(3)
    P0 = np.eye(6)
    Q = np.eye(6) * 0.01
    R = np.eye(3) * 10
    #Q = np.eye(6) * 1
    #R = np.eye(3) * 1
    measurements = measure

    def calculate_observation_covariance(measurements):
        measurements = [measurements[i][0:3] for i in range(len(measurements))]
        mean_measurements = np.mean(measurements, axis=0)
        errors = measurements - mean_measurements
        R = np.cov(errors, rowvar=False)
        return R
    
    R_ = calculate_observation_covariance(measurements)
    x0[:3] = np.array(measurements[0][0:3])
    
    dt = 1  # 假设时间步长为 1，可以根据实际情况调整
    predicted_states, updated_states = unscented_kalman_filter(x0, P0, Q, R, measurements, process_model, measurement_model, dt)
    
    #return updated_states
    
    return[state[:3] for state in updated_states]

def GPY(json_file):
    # 读取数据和存数据部分
    with open(json_file, newline='') as jsonfile:
        data = json.load(jsonfile)
        temp_all = []
        for key in range(33):
            temp_thiskey_x = []
            temp_thiskey_y = []
            temp_thiskey_z = []
            visib = []
            this_joint = []
            for i in range(len(data)):
                pos = []
                pos.append(data[i]['keypoints'][key][1])
                pos.append(data[i]['keypoints'][key][2])
                pos.append(data[i]['keypoints'][key][3])
                pos.append(data[i]['keypoints'][key][4])
                this_joint.append(pos)
            returnx = UKFmain(this_joint)
            for j in range(len(returnx)):
                here = [key, returnx[j][0], returnx[j][1], returnx[j][2], this_joint[j][3]]
                temp_all.append(here)

        dict_all = []  # 放所有dict的地方，想把它弄成跟之前的格式一样
        
        for frame in range(len(data)):
            temp_dic = {'frame': frame + 1, 'keypoints': []}
            for lll in range(len(temp_all)):
                if frame == lll % len(data):
                    temp_dic['keypoints'].append(temp_all[lll])
            dict_all.append(temp_dic)

        output_json = json_file.split('.')[0] + '_UKFA_picQ001R10A12.json'
        write_json(dict_all, output_json)


if __name__ == "__main__":

    json_file = "E://things/master/pose3d/result/S001C001P001R001A038_rgb/S001C001P001R001A038_rgb_sharpenEMA.json"
    #json_file = "E://things/master/pose3d/result/crawl0301/crawl0301EMA.json"
    #json_file = "E://things/master/pose3d/result/TC_S1_freestyle1_cam1/TC_S1_freestyle1_cam1EMA.json" #計算用
    #json_file = "E://things/master/pose3d/result/TC_S1_freestyle1_cam1/freestyle1900_chEMA.json"
    #json_file ="E://things/master/pose3d/result/TC_S1_acting1_cam1/acting2900_chEMA.json"
    #json_file = "E://things/master/pose3d/result/myData0301-1/myData0301-1EMA.json"
    #json_file = input("json:")  # 呼叫用
    GPY(json_file)
