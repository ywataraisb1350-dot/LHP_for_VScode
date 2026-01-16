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

def ec_flat(Tec, Tev):
    Q_ev = d.alpha* d.A_wick* (Tec- Tev)
    M_dot = Q_ev/ p.lambda_lv(Tev)
    Delta_L = d.L_gr/ d.num_cal_ec
    u_max_times_rho = M_dot/(d.w_gr* d.h_gr* d.n_gr)
    
    Q_gr, P_loss_gr = 0, 0
    u, P, T = 0, p.P_sat(Tev), Tev
    #print('u=',u, 'P=',P, 'T=',T, 'd_gr',d.d_gr, 'Re=',p.Re_g(u, P, T, d.d_gr))
    ec_result = []
    current_data = {
            'step': 'start',
            'distance':'start',
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho_g': p.rho_g(P,T),
            'm_dot':M_dot* (Delta_L)/ (d.n_gr* d.L_wick),
            'Re':p.Re_g(u, P, T, d.d_gr),
            'h_g':p.h_g(u, P, T, d.d_gr),
            'tau_g':p.tau_g(u, P, T, d.d_gr),
            'Q_gr':Q_gr,
            'Sigma_Plos_gr':P_loss_gr,
            'A_recieve_heat':Delta_L* d.w_gr+ 2* Delta_L* d.h_gr
            }
    ec_result.append(current_data)
    
    for i in range(d.num_cal_ec+ 1):
        print('u=',u, 'P=',P, 'T=',T, 'd_gr',d.d_gr, 'Re=',p.Re_g(u, P, T, d.d_gr), 'i',i)
        
        T_next = ( (p.h_g(u, P, T, d.d_gr)*(Delta_L* d.w_gr+ 2* Delta_L* d.h_gr)*(Tec- Tev)
             + (M_dot*i*Delta_L* p.Cp_g(T)* T/ (d.n_gr* d.L_wick))+ (M_dot*Delta_L* p.Cp_g(Tev)* Tev/ (d.n_gr* d.L_wick)))
             / (  (M_dot*i*Delta_L* p.Cp_g(T)/ (d.n_gr* d.L_wick))+ (M_dot*Delta_L* p.Cp_g(Tev)/ (d.n_gr* d.L_wick)) ))
        
        u_max = u_max_times_rho/ p.rho_g(P, T)
        P_next = P- 4* p.tau_g(u_max, P, T, d.d_gr)* Delta_L/ d.d_gr
        
        u = M_dot* (i+ 1)* Delta_L/(d.n_gr* d.L_wick* p.rho_g(P_next, T_next)* d.w_gr* d.h_gr)
        
        Q_gr = Q_gr+ (M_dot*i*Delta_L* p.Cp_g(T)* (T_next- T)/ (d.n_gr* d.L_wick))+ (M_dot*Delta_L* p.Cp_g(Tev)* (T_next- Tev)/ (d.n_gr* d.L_wick))* d.n_gr
        P_loss_gr = P_loss_gr+ P- P_next
        
        P, T = P_next, T_next
        current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho_g':p.rho_g(P,T),
            'm_dot':M_dot* (i* Delta_L)/ (d.n_gr* d.L_wick),
            'Re':p.Re_g(u, P, T, d.d_gr),
            'mu_g':p.mu_g(T),
            'h_g':p.h_g(u, P, T, d.d_gr),
            'tau_g':p.tau_g(u, P, T, d.d_gr),
            'Q_gr':Q_gr,
            'Sigma_Plos_gr':P_loss_gr,
            'A_recieve_heat':Delta_L* d.w_gr+ 2* Delta_L* d.h_gr
            }
        ec_result.append(current_data)
    df_ec = pd.DataFrame(ec_result)

    P_cap = 2*p.sigma(Tev)* math.cos(d.contact_angle)/d.r_max_pore
    P_loss_wick_flat = p.mu_l(Tev)* (d.H_wick- d.h_gr) * M_dot/(d.K_wick* d.A_wick* p.rho_l(Tev))
    P_loss_wick_gr = p.mu_l(Tev)* d.h_gr* M_dot/(d.K_wick* (d.A_wick- d.w_gr* d.L_gr* d.n_gr) * p.rho_l(Tev))

    return P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_gr, P_loss_wick_flat, P_loss_wick_gr, P_cap