import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
from scipy.interpolate import interp1d
from datetime import datetime

import design_prop

design_dict = design_prop.design()
d = types.SimpleNamespace(**design_dict)
prop_dict = design_prop.prop(d.csv_path, d.csv_path_inv)
p = types.SimpleNamespace(**prop_dict)

def h_con_Traviss(u, T, d, X_LM):
    Re_l_res = p.Re_l(u,T,d)
    Pr_l_res = p.Pr_l(T)
    if Re_l_res <= 50:
        Ft = 0.707* Pr_l_res* Re_l_res**0.5
        
    elif 50 <= Re_l_res <= 1125:
        Ft = 5* Pr_l_res+ 5*math.log10(1+ Pr_l_res* (0.0964* Re_l_res**(0.585) - 1))
        
    elif 1125 <= Re_l_res:
        Ft = 5* Pr_l_res+ 5* math.log10(1+ 5* Pr_l_res)+ 2.5* math.log10(0.0031* Re_l_res**(0.812))
        
    h_in = p.k_l(T)* 0.15* Pr_l_res* (Re_l_res**0.9)* (1/X_LM + 2.85/(X_LM**0.476)) / (d* Ft)
    
    return h_in

def G_gas(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex):
    h_in = p.h_g(u,P,T,d_i)
    #h_ex = (T-T_ex)*?
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_gas = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_gas, 1/R_1, 1/R_2, R_3, 1/R_4

def G_liq(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex):
    h_in = p.h_l(u, P, T, d_i)
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_liq = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_liq, 1/R_1, 1/R_2, R_3, 1/R_4

def G_mix(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex, M_dot, x, X_LM):
    #h_in = beta
    #h_in = h_con(u, M_dot, x, P, T, d_i)
    h_in = h_con_Traviss(u, T, d_i, X_LM)
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_mix = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_mix, 1/R_1, 1/R_2, R_3, 1/R_4, h_in

def trans_line(u, P, T, rho, x, phase, M_dot, L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex, num_cal):
    Delta_L = L/ num_cal
    Q_release_gas, Q_release_mix, Q_release_liq = 0, 0, 0
    P_loss_gas, P_loss_mix, P_loss_liq, P_loss_all_phase = 0, 0, 0, 0
    result = []
    u = 4* M_dot/ (rho* math.pi* d_i**2)
    P_ini = P
    T_ini = T
    
    current_data = {
            'step': 'start',
            'distance':'start',
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho': rho,
            'm_dot':u*rho*math.pi*0.25*d_i**2
            }
    result.append(current_data)
    
    for i in range(num_cal+ 1):
        
        if phase=='gas':
            G_gas_val, G_in, G_pipe, R_insu, G_ex = G_gas(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            T_next = T- G_gas_val* (T- T_ex)/ (M_dot* p.Cp_g(T))
            P_next = P- 4* p.tau_g(u, P, T, d_i)* Delta_L/ d_i
            P_loss_gas = P_loss_gas+ (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_gas = Q_release_gas+ G_gas_val* (T- T_ex)
            Delta_T = T-T_next
            P, T = P_next, T_next
            rho = p.rho_g(P, T)
            u = 4* M_dot/ (rho* math.pi* d_i**2)
                
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho': rho,
            'phase':phase,
            'x':x,
            'm_dot':u*rho*math.pi*0.25*d_i**2,
            'P_loss_gas':P_loss_gas,
            "P_loss_all_phase":P_loss_all_phase,
            'Q_release_gas':Q_release_gas,
            'Delta_T':Delta_T,
            'G_in_gas':G_in,
            'G_pipe_gas':G_pipe,
            'R_insu_gas':R_insu,
            'G_ex_gas':G_ex
            }
            result.append(current_data)
            
            if P>= p.P_sat(T):
                #P = P_sat(T)
                x = 0.999999999
                phase = 'mix'
            
        elif phase=='mix':
            
            Delta_P_2p_val, C, X_LM = p.Delta_P_2p(u, T, d_i, x, M_dot, Delta_L, P)
            if X_LM<1e-3:
                X_LM = 1e-3
            G_mix_val, G_in, G_pipe, R_insu, G_ex, h_in = G_mix(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex, M_dot, x, X_LM)
            G_gas_val, G_in, G_pipe, R_insu, G_ex = G_gas(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            G_liq_val, G_in, G_pipe, R_insu, G_ex = G_liq(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            '''
            if 0.9 < x <=1.5:
                G = G_gas_val
            elif 0.1 <= x <= 0.9:
                G = G_mix_val
            elif x < 0.1:
                G = G_liq_val
            '''
            
            P_next = P- Delta_P_2p_val
            T_next = p.T_sat(P)
            G = G_mix_val
            x = x+ (x* p.Cp_g(T)+ (1-x)* p.Cp_l(T))* (T- T_next)/ p.lambda_lv(T)- G* (T- T_ex)/ (M_dot* p.lambda_lv(T))
            if x>0.999999999:
                phase, x = 'gas', 1
                
            elif x <= 1e-6:
                phase, x = 'liq', 0
            u = max([ (M_dot* x/ (p.rho_g(P_next, T_next)* math.pi* 0.25* d_i**2) ), ( M_dot/ (p.rho_l(T_next)* math.pi* 0.25* d_i**2) ) ])
            
            P_loss_mix = P_loss_mix + (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_mix = Q_release_mix + G_mix_val* (T- T_ex)
            Delta_T = T- T_next
            P, T = P_next, T_next
            rho = p.rho_g(P, T) #prop(T, 'rho_g', all_funcs)
            
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho': '-',
            'phase':phase,
            'x':x,
            'm_dot':x*u*p.rho_g(P,T)*math.pi*0.25*d_i**2,
            "h_in":h_in,
            'P_loss_mix':P_loss_mix,
            'LM_C':C,
            'LM_X':X_LM,
            'Delta_P_2p':Delta_P_2p_val,
            "P_loss_all_phase":P_loss_all_phase,
            'Q_release_mix':Q_release_mix,
            'Delta_T':Delta_T,
            'G_in_mix':G_in,
            'G_pipe_mix':G_pipe,
            'R_insu_mix':R_insu,
            'G_ex_mix':G_ex
            }
            result.append(current_data)
                
        elif phase=='liq':
            G_liq_val, G_in, G_pipe, R_insu, G_ex = G_liq(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            T_next = T- G_liq_val* (T- T_ex)/ (M_dot* p.Cp_l(T))
            P_next = P- 4* p.tau_l(u, P, T, d_i)* Delta_L/ d_i
            P_loss_liq = P_loss_liq+ (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_liq = Q_release_liq+ G_liq_val* (T- T_ex)
            Delta_T = T-T_next
            P, T = P_next, T_next
            rho = p.rho_l(T)
            u = 4* M_dot/ (rho* math.pi* d_i**2)
                
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho': rho,
            'phase':phase,
            'x':x,
            'm_dot':u*rho*math.pi*0.25*d_i**2,
            'P_loss_liq':P_loss_liq,
            'P_loss_all_phase':P_loss_all_phase,
            'Q_release_liq':Q_release_liq,
            'Delta_T':Delta_T,
            'G_in_liq':G_in,
            'G_pipe_liq':G_pipe,
            'R_insu_liq':R_insu,
            'G_ex_liq':G_ex
            }
            result.append(current_data)
            
    P_loss = P_ini- P
    #P_loss = P_loss_all_phase
    T_ave = (T_ini+ T)/2
    df_line = pd.DataFrame(result)

    return u, P, T, rho, x, phase, df_line, P_loss, T_ave, T_ini