import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
from scipy.interpolate import interp1d
from datetime import datetime

import design_prop
import ec_flat
import transline

epsilon = 1e-6
random_start_Tev_min, random_start_Tev_max = 42.09660727333085 +273.15, 42.09660727333085 +273.15000000001
random_start_deltat_min, random_start_deltat_max =(47.676987038329-42.09660727333085), (47.676987038329-42.0966072733308)
max_restarts = 100
iterations = 30000
learning_ratio = 2e-2
grad_clip_threshold = 50000
learning_rate_adam = 0.02 # 固定学習率より少し大きめに設定できることが多い
beta1 = 0.9
beta2 = 0.9
epsilon_adam = 1e-5
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

def eval_func(Tec, Tev):

    P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_gr, P_loss_wick_flat, P_loss_wick_gr, P_cap = ec_flat.ec_flat(Tec, Tev)
    P_loss_wick = P_loss_wick_flat+ P_loss_wick_gr

    rho = p.rho_g(P,T)
    x, phase = 1, 'gas'
    u = 4* M_dot/ (p.rho_g(P,T)* math.pi* d.d_i_vl **2)

    u, P, T, rho, x, phase, df_vl, P_loss_vl, T_ave_vl, T_ini_vl = transline.trans_line(
        u, P, T, rho, x, phase, M_dot, d.L_vl, d.d_i_vl, d.d_o_vl, d.d_o_insu_vl, d.k_vl, d.k_insu_vl, d.h_out, d.T_amb, d.num_cal_vl)

    u, P, T, rho, x, phase, df_cl, P_loss_cl, T_ave_cl, T_ini_cl = transline.trans_line(
        u, P, T, rho, x, phase, M_dot/2, d.L_cl, d.d_i_cl, d.d_o_cl, d.d_o_insu_cl, d.k_cl, d.k_insu_cl, d.h_sink, d.T_sink, d.num_cal_cl)

    u, P, T, rho, x, phase, df_ll, P_loss_ll, T_ave_ll, T_ini_ll = transline.trans_line(
        u, P, T, rho, x, phase, M_dot, d.L_ll, d.d_i_ll, d.d_o_ll, d.d_o_insu_ll, d.k_ll, d.k_insu_ll, d.h_out, d.T_amb, d.num_cal_ll)
    
    T_ccin = p.T_sat(P)
    G_ec_ccc = d.k_cc_flange* (d.w_flange* d.l_flange- d.w_flange_hole- d.l_flange_hole)/ d.L_ccpipe
    G_ccc_ccin = 4.36* p.k_l(T_ccin)* (2* math.pi* d.r_cc* d.H_cc)/ d.d_e_cc + 4.36* p.k_l(T_ccin)* math.pi* d.r_cc* 0.25

    T_ccc =( (G_ccc_ccin* T_ccin + d.h_out* (math.pi* d.r_cc**2 *  0.25+ 2* math.pi* d.r_cc* d.H_cc)* d.T_amb + G_ec_ccc* Tec)/
        (G_ccc_ccin+ d.h_out* (math.pi* d.r_cc**2 *  0.25+ 2* math.pi* d.r_cc* d.H_cc) + G_ec_ccc) )
    
    k_eff = d.epsilon_wick* p.k_l(T_ccin)+ (1- d.epsilon_wick)* d.k_wick
    Q_ec_wick_ccin = 3* k_eff* d.A_wick* (Tev- T_ccin)/ d.H_wick
    Q_ec_ccc = G_ec_ccc* (Tec- T_ccc)
    Q_ec_amb = d.h_out* (d.W_ec* d.L_ec- d.A_hs + d.H_ec*d.W_ec*2 + d.H_ec*d.L_ec*2)* (Tec- d.T_amb)
    T_hs = (d.Q_load + d.h_out* d.A_hs* d.T_amb + d.h_hs_ec* d.W_ec*d.L_ec * Tec)/(d.h_out* d.A_hs + d.h_hs_ec* d.W_ec*d.L_ec)
    #Q_hs_amb = d.h_out* d.A_hs* (T_hs- T_amb)
    Q_ec_in = d.Q_load #- Q_hs_amb
    Q_ec_out = Q_ev+ Q_gr+ Q_ec_ccc+ Q_ec_wick_ccin+ Q_ec_amb
    eval_ec = (100*(Q_ec_in- Q_ec_out)/ d.Q_load)**2
  
    Q_cc_ll = M_dot* p.Cp_l(T)* (T_ccin- T)
    Q_ccc_ccin = G_ccc_ccin(T_ccin)* (T_ccc-T_ccin)
    eval_cc = (100*(Q_ccc_ccin+ Q_ec_wick_ccin- Q_cc_ll)/ (d.Q_load))**2

    result_dict={
        "Tec":Tec,
        "Tev":Tev,
        "T_ini_vl":T_ini_vl,
        "T_ave_vl":T_ave_vl,
        "T_ini_cl":T_ini_cl,
        "T_ave_cl":T_ave_cl,
        "T_ini_ll":T_ini_ll,
        "T_ave_ll":T_ave_ll,
        "T_ccin":T_ccin,
        "T_ccc":T_ccc,
        "T_hs":T_hs,
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

    return eval_ec+eval_cc, df_ec, df_vl, df_cl, df_ll, 

