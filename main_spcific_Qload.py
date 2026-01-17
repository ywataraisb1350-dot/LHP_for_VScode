import math
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
from scipy.interpolate import interp1d
from datetime import datetime

import design_prop
import ec_flat
import transline

Q_load = 5600 

epsilon = 1e-6
random_start_Tev_min, random_start_Tev_max = 40+273.15, 50.09660727333085 +273.15000000001
random_start_deltat_min, random_start_deltat_max =(47.676987038329-42.09660727333085), (70.676987038329-42.0966072733308)
max_restarts = 100
iterations = 30000
learning_ratio = 2e-2
grad_clip_threshold = 50000
learning_rate_adam = 0.2 # 固定学習率より少し大きめに設定できることが多い
beta1 = 0.9
beta2 = 0.9
epsilon_adam = 0.1
m_t = np.zeros(2) # モーメントベクトル
v_t = np.zeros(2)

dict_cal_parameter = {
    "epsilon":epsilon,
    "random_start_Tev_min":random_start_Tev_min,
    "random_start_Tev_max":random_start_Tev_max,
    "random_start_deltaT_min":random_start_deltat_min,
    "random_start_deltaT_max":random_start_deltat_max,
    "max_restarts":max_restarts,
    "iterations":iterations,
    "learning_ratio":learning_ratio,
    "grad_clip_threshold":grad_clip_threshold,
    "learning_rate_adam":learning_rate_adam,
    "beta1":beta1,
    "beta2":beta2,
    "epsilon_adam":epsilon_adam
}

design_dict = design_prop.design()
d = types.SimpleNamespace(**design_dict)
prop_dict = design_prop.prop(d.csv_path, d.csv_path_inv)
p = types.SimpleNamespace(**prop_dict)

def eval_func(Tec, Tev, Q_load):

    P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_gr, P_loss_wick_flat, P_loss_wick_gr, P_cap = ec_flat.ec_flat(Tec, Tev)
    P_loss_wick = P_loss_wick_flat+ P_loss_wick_gr

    rho = p.rho_g(P,T)
    x, phase = 1, 'gas'
    u = 4* M_dot/ (p.rho_g(P,T)* math.pi* d.d_i_vl **2)

    u, P, T, rho, x, phase, df_vl, P_loss_vl, T_ave_vl, T_ini_vl = transline.trans_line(
        u, P, T, rho, x, phase, M_dot, d.L_vl, d.d_i_vl, d.d_o_vl, d.d_o_insu_vl, d.k_vl, d.k_insu_vl, d.h_out, d.T_amb, d.num_cal_vl)

    u, P, T, rho, x, phase, df_cl, P_loss_cl, T_ave_cl, T_ini_cl = transline.trans_line(
        u, P, T, rho, x, phase, M_dot/2, d.L_cl, d.d_i_cl, d.d_o_cl, d.d_o_cl, d.k_cl, d.k_insu_cl, d.h_sink, d.T_sink, d.num_cal_cl)

    u, P, T, rho, x, phase, df_ll, P_loss_ll, T_ave_ll, T_ini_ll = transline.trans_line(
        u, P, T, rho, x, phase, M_dot, d.L_ll, d.d_i_ll, d.d_o_ll, d.d_o_insu_ll, d.k_ll, d.k_insu_ll, d.h_out, d.T_amb, d.num_cal_ll)
    
    T_ccin = p.T_sat(P)

    G_ec_ccc = d.k_flange* (d.w_flange* d.l_flange- d.w_flange_hole* d.l_flange_hole)/ d.L_ccpipe
    G_ccc_ccin = 4.36* p.k_l(T_ccin)* (2* math.pi* d.r_cc* d.H_cc)/ d.d_e_cc + 4.36* p.k_l(T_ccin)* math.pi* d.r_cc* 0.25

    T_ccc =( (G_ccc_ccin* T_ccin + d.h_out* (math.pi* d.r_cc**2 *  0.25+ 2* math.pi* d.r_cc* d.H_cc)* d.T_amb + G_ec_ccc* Tec)/
        (G_ccc_ccin+ d.h_out* (math.pi* d.r_cc**2 *  0.25+ 2* math.pi* d.r_cc* d.H_cc) + G_ec_ccc) )
    
    k_eff = d.epsilon_wick* p.k_l(T_ccin)+ (1- d.epsilon_wick)* d.k_wick
    Q_ec_wick_ccin = 3* k_eff* d.A_wick* (Tev- T_ccin)/ d.H_wick
    Q_ec_ccc = G_ec_ccc* (Tec- T_ccc)
    Q_ec_amb = d.h_out* (d.W_ec* d.L_ec- d.A_hs + d.H_ec*d.W_ec*2 + d.H_ec*d.L_ec*2)* (Tec- d.T_amb)
    T_hs = Tec+ Q_load/(d.h_hs_ec* d.A_hs)
    Q_ec_in = Q_load #- Q_hs_amb
    Q_ec_out = Q_ev+ Q_gr+ Q_ec_ccc+ Q_ec_wick_ccin+ Q_ec_amb
    eval_ec = (100*(Q_ec_in- Q_ec_out)/ Q_load)**2
  
    Q_cc_ll = M_dot* p.Cp_l(T)* (T_ccin- T)
    Q_ccc_ccin = G_ccc_ccin* (T_ccc-T_ccin)
    eval_cc = (100*(Q_ccc_ccin+ Q_ec_wick_ccin- Q_cc_ll)/ Q_load)**2

    result_dict={
        "Tec":Tec-273.15,
        "Tev":Tev-273.15,
        "T_ini_vl":T_ini_vl-273.15,
        "T_ave_vl":T_ave_vl-273.15,
        "T_ini_cl":T_ini_cl-273.15,
        "T_ave_cl":T_ave_cl-273.15,
        "T_ini_ll":T_ini_ll-273.15,
        "T_ave_ll":T_ave_ll-273.15,
        "T_ccin":T_ccin-273.15,
        "T_ccc":T_ccc-273.15,
        "T_hs":T_hs-273.15,
        
        "Q_load":Q_load,
        "Q_ev":Q_ev,
        "Q_gr":Q_gr,
        "Q_ec_wick_ccin":Q_ec_wick_ccin,
        "Q_ec_ccc":Q_ec_ccc,
        "Q_ec_amb":Q_ec_amb,
        #"Q_hs_amb"
        "Q_cc_ll":Q_cc_ll,
        "Q_ccc_ccin":Q_ccc_ccin,
        "eval_ec[%]":math.sqrt(eval_ec),
        "eval_cc[%]":math.sqrt(eval_cc),
        
        "P_cap.":P_cap,
        "P_loss_wick":P_loss_wick,
        "P_loss_gr":P_loss_gr,
        "P_loss_vl":P_loss_vl,
        "P_loss_cl":P_loss_cl,
        "P_loss_ll":P_loss_ll
    }

    return (eval_ec+eval_cc, df_ec, df_vl, df_cl, df_ll, 
            T_hs, Tec, T_ave_cl, P_cap, P_loss_wick, P_loss_gr, P_loss_vl, P_loss_cl, P_loss_ll, result_dict)
    
key_result = []
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(timestamp, exist_ok=True)

for restart_count in range(max_restarts):
        Tev = random.uniform(random_start_Tev_min, random_start_Tev_max)
        Tec = random.uniform(Tev + random_start_deltat_min, Tev + random_start_deltat_max)
        current_pos = np.array([Tec, Tev])
        print(f"\nリスタート{restart_count+1}")
        print(f"\n新しい初期値: Tec={current_pos[0]-273.15}, Tev={current_pos[1]-273.15}")
        local_min_val = [None, None, float('inf')]
        violation_found = False
        
        for i in range(iterations):
            eval_val_current = eval_func(Tec, Tev, Q_load)[0]
            
            grad = np.array([((eval_func(Tec+ epsilon, Tev, Q_load)[0] - eval_func(Tec- epsilon, Tev, Q_load)[0] ) / (2* epsilon)),
                             ((eval_func(Tec, Tev+ epsilon, Q_load)[0] - eval_func(Tec, Tev- epsilon, Q_load)[0] ) / (2* epsilon))])
            grad_norm = np.linalg.norm(grad)    
            if grad_norm > grad_clip_threshold:
                grad = grad / grad_norm * grad_clip_threshold
                
            t = i + 1
            m_t = beta1 * m_t + (1 - beta1) * grad      # モーメントの更新
            v_t = beta2 * v_t + (1 - beta2) * (grad**2) # 2次モーメントの更新
            m_hat = m_t / (1 - beta1**t) # バイアス補正
            v_hat = v_t / (1 - beta2**t) # バイアス補正
            update_vector = learning_rate_adam * m_hat / (np.sqrt(v_hat) + epsilon_adam)
            next_pos_candidate = current_pos - update_vector
            
            Tec_candidate = next_pos_candidate[0]
            Tev_candidate = next_pos_candidate[1]
            eval_val_next = eval_func(Tec_candidate, Tev_candidate, Q_load)[0]
            
            if np.isnan(next_pos_candidate).any():
                print(f"\nステップ {i+1} で値がnanになりました")
                violation_found = True
                break
            
            if next_pos_candidate[1] <= 300 or next_pos_candidate[0] <= next_pos_candidate[1]:
                print(f"\nステップ {i+1} で制約違反。", 'T_ec=', next_pos_candidate[0], 'T_ev=', next_pos_candidate[1])
                violation_found = True
                break
            
            current_pos = next_pos_candidate
            Tec = current_pos[0]
            Tev = current_pos[1]
            eval_val_current = eval_val_next
            
            if eval_val_current < 0.5:
                print(f"\nステップ {i+1} で評価関数の値が {eval_val_current} となり、0.5未満になったため計算を終了します。")
                print('Tec=', Tec_candidate-273.15, 'Tev=', Tev_candidate-273.15)
                global_min_val = [Tec, Tev, eval_val_current]
                convergence = True
                break
            
            if(i+ 1)%5 == 0:
                print('Tec=', Tec-273.15, 'Tev=', Tev-273.15, 'eval_func=', eval_val_current)
                print('step', i+1)
                
            if eval_val_current < local_min_val[2]:
                local_min_val = [Tec, Tev, eval_val_current]
                
        if convergence:
            break
        
        if local_min_val[0] is not None:
            if local_min_val[2] < global_min_val[2]:
                global_min_val = local_min_val
                
(eval_val, df_ec, df_vl, df_cl, df_ll, T_hs, Tec, T_ave_cl, P_cap, P_loss_wick, 
     P_loss_gr, P_loss_vl, P_loss_cl, P_loss_ll, result_dict) = eval_func(global_min_val[0], global_min_val[1], Q_load)
    
status_str = "True" if convergence else "False"
sub_dir_name = f"{timestamp}_{Q_load}W"
sub_dir = os.path.join(timestamp, sub_dir_name)
os.makedirs(sub_dir, exist_ok=True)
file_path_ec = os.path.join(sub_dir, f'ec_{timestamp}_{Q_load}W_{status_str}.csv')
df_ec.to_csv(file_path_ec, index=False)
file_path_vl = os.path.join(sub_dir, f'vl_{timestamp}_{Q_load}W_{status_str}.csv')
df_vl.to_csv(file_path_vl, index=False)
file_path_cl = os.path.join(sub_dir, f'cl_{timestamp}_{Q_load}W_{status_str}.csv')
df_cl.to_csv(file_path_cl, index=False)
file_path_ll = os.path.join(sub_dir, f'll_{timestamp}_{Q_load}W_{status_str}.csv')
df_ll.to_csv(file_path_ll, index=False)
df_res = pd.DataFrame(result_dict.items(), columns=['lavel', 'val'])
file_path_res = os.path.join(sub_dir, f'result_{timestamp}_{Q_load}W_{status_str}.csv')
df_res.to_csv(file_path_res, index=False)
    
key_result_dict = {
        "Q_load":Q_load,
        "HS":T_hs-273.15,
        "EC":Tec-273.15,
        "CL ave.":T_ave_cl-273.15,
        "wick":P_loss_wick,
        "groove":P_loss_gr,
        "VL":P_loss_vl,
        "CL":P_loss_cl,
        "LL":P_loss_ll,
        "Cap.":P_cap,
        "converg_judge":convergence
        }
    
key_result.append(key_result_dict)
    
df_keyres = pd.DataFrame(key_result)
file_path_keyres = os.path.join(timestamp, f'KEYresult_{timestamp}.csv')
df_keyres.to_csv(file_path_keyres, index=False)

df_cal_para = pd.DataFrame(list(dict_cal_parameter.items()), columns=['parameter', 'value'])
file_path_cal_para = os.path.join(timestamp, f'cal_para_{timestamp}.csv')
df_cal_para.to_csv(file_path_cal_para, index=False)

df_design = pd.DataFrame(list(design_dict.items()), columns=['parameter', 'value'])
file_path_design = os.path.join(timestamp, f'design_{timestamp}.csv')
df_design.to_csv(file_path_design, index=False)