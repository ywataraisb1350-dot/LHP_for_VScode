import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime

def design():
    #design parameter input zone
    Q_load = 5600   #[W]
    csv_path = 'R1233zdE.csv'
    csv_path_inv = 'R1233zdE_inv.csv'

    W_ec = 320  #mm
    L_ec = 320  #mm
    t_ec_bt = 5 #mm
    t_ec = 10   #mm
    t_ec_up = 5 #mm
    k_ec = 390  #conductivity
    H_ec = 40   #mm

    r_cc = 160
    t_cc_bt = 4
    t_cc_up = 4
    t_cc = 4
    H_cc = 200

    k_ec_flange = 398
    k_cc_flange = 16
    t_ec_flange = 4
    t_cc_flange = 4
    n_flange = 4

    A_ec_btm_up = 20000 #mm^2 

    W_wick_btm = 280
    L_wick_btm = 280
    H_wick_btm = 8          #=thickness mm include groove
    k_wick_btm = 16         #[W/m-K]
    r_max_pore_btm = 3      #[micro m] not diameter , enter radius
    epsilon_wick_btm = 0.6  #[-]
    K_wick_btm = 3.2e-13      #[m^2]
    contact_angle_btm = 10  #[deg]
    n_gr_btm = 176           #num of groove
    w_gr_btm = 3
    h_gr_btm = 3
    L_gr_btm = 62 

    W_wick_up = 280
    L_wick_up = 280
    H_wick_up = 8          #=thickness mm include groove
    k_wick_up = 16         #[W/m-K]
    r_max_pore_up = 3      #[micro m] not diameter , enter radius
    epsilon_wick_up = 0.6  #[-]
    K_wick_up = 3.2e-13      #[m^2]
    contact_angle_up = 10  #[deg]
    n_gr_up = 176           #num of groove
    w_gr_up = 3
    h_gr_up = 3
    L_gr_up = 62

    d_i_vl = 27.6
    d_o_vl = 31.8
    L_vl = 15           #[m]
    t_insu_vl = 50      #thickness mm
    k_vl = 16           #[W/m-K]
    k_insu_vl = 0.004   #[W/m-K]

    d_i_cl = 16.5
    d_o_cl = 19.1
    L_cl = 15           #[m]
    k_cl = 16
    k_insu_cl = 1000    #実際はinsuなし，計算の便宜を図るため導入

    d_i_ll = 16.5
    d_o_ll = 19.1
    L_ll = 15           #[m]
    t_insu_ll = 50      #thickness mm
    k_ll = 16           #[W/m-K]
    k_insu_ll = 0.004   #[W/m-K]

    T_amb = 30      #[celsius temp]
    T_sink = 30
    alpha = 2500   #[W/m^2-K] 蒸発熱伝達率A_ecベースの値
    h_hs_ec = 4000  #[W/m^2-K] 熱源-蒸発器熱伝達率
    h_out = 20.0    #[W/m^2-K] 決め打ち外部への放熱伝達率
    h_sink = 800.0  #[W/m^2-K] 決め打ち外部へのコンデンサ放熱伝達率
    grav_ac = 9.8   #gravity_acceralation

    num_cal_ec, num_cal_vl, num_cal_cl, num_cal_ll = 100, 100, 200, 100
    #input zone end

    #design para convert to SI unit
    W_ec = W_ec*1e-3
    L_ec = L_ec*1e-3
    t_ec_bt = t_ec_bt*1e-3
    t_ec = t_ec*1e-3
    t_ec_up = t_ec_up*1e-3
    H_ec = H_ec*1e-3

    r_cc = r_cc*1e-3
    t_cc_bt = t_cc_bt*1e-3
    t_cc_up = t_ec_up*1e-3
    t_cc = t_cc*1e-3
    H_cc = H_cc*1e-3

    t_ec_flange = t_ec_flange*1e-3
    t_cc_flange = t_cc_flange*1e-3
    t_kubire = t_kubire*1e-3
    w_kubire = w_kubire*1e-3
    l_kubire = l_kubire*1e-3

    W_wick = 280
    L_wick = 280
    H_wick = 8
    k_wick = 16         #[W/m-K]
    r_max_pore = 3      #[micro m] not diameter , enter radius
    epsilon_wick = 0.6  #[-]
    K_wick = 3.2e-13      #[m^2]
    contact_angle = 10  #[deg]
    n_gr = 184           #num of groove
    w_gr = 3
    h_gr = 3
    L_gr = 70 

    d_i_vl = 27.6
    d_o_vl = 31.8
    L_vl = 15           #[m]
    t_insu_vl = 50      #thickness mm
    k_vl = 16           #[W/m-K]
    k_insu_vl = 0.004   #[W/m-K]

    d_i_cl = 16.5
    d_o_cl = 19.1
    L_cl = 15           #[m]
    k_cl = 16
    k_insu_cl = 1000    #実際はinsuなし，計算の便宜を図るため導入

    d_i_ll = 16.5
    d_o_ll = 19.1
    L_ll = 15           #[m]
    t_insu_ll = 50      #thickness mm
    k_ll = 16           #[W/m-K]
    k_insu_ll = 0.004   #[W/m-K]

    T_amb = 30      #[celsius temp]
    T_sink = 30

    design_dict = {
           "W_ec":W_ec,
           "L_ec":L_ec,
           "csv_path":csv_path,
           "csv_path_inv":csv_path_inv

    }

    return design_dict

def create_interpolation_functions(csv_path):
    df = pd.read_csv(csv_path)
    
    x_col_name = df.columns[0]
    x_data = df[x_col_name].values
    interpolation_funcs = {}
    
    for y_col_name in df.columns[1:]:
        y_data = df[y_col_name].values
        f = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolation_funcs[y_col_name] = f
        
    return interpolation_funcs

def prop_all(T, label, funcs_dict):
    if label not in funcs_dict:
        print(f"Error: Label '{label}' not found in the CSV data.")
        return None
    
    target_function = funcs_dict[label]
    prop_array = target_function(T)
    prop_scalar = prop_array.item()
    
    return prop_scalar

def prop(csv_path, csv_path_inv):
    all_funcs = create_interpolation_functions(csv_path)
    sat_inv = create_interpolation_functions(csv_path_inv)

    def P_sat(T):
        P_sat = prop_all(T, 'P_sat', all_funcs)
        return P_sat
    
    def rho_l(T):
        rho_l = prop_all(T, 'rho_l', all_funcs)
        return rho_l
    
    def rho_g(P, T):
        #rho_g = P* 0.1305/ (8.311* T)
        rho_g = prop_all(T, 'rho_g', all_funcs)
        return rho_g
    
    def Cp_l(T):
        Cp_l = prop(T, 'Cp_l', all_funcs)
        return Cp_l

    def Cp_g(T):
        Cp_g = prop(T, 'Cp_g', all_funcs)
        return Cp_g

    def lambda_lv(T):
        lambda_lv = prop(T, 'lambda', all_funcs)
        return lambda_lv

    def k_l(T):
        k_l = prop(T, 'k_l', all_funcs)
        return k_l

    def k_g(T):
        k_g = prop(T, 'k_g', all_funcs)
        return k_g

    def mu_l(T):
        mu_l = prop(T, 'mu_l', all_funcs)
        return mu_l

    def mu_g(T):
        mu_g = prop(T, 'mu_g', all_funcs)
        return mu_g

    def Pr_l(T):
        Pr_l = mu_l(T)* Cp_l(T)/ k_l(T)
        return Pr_l

    def Pr_g(T):
        Pr_g = mu_g(T)* Cp_g(T)/ k_g(T)
        return Pr_g

    def nu_l(T):
        nu_l = mu_l(T)/ rho_l(T)
        return nu_l

    def nu_g(P, T):
        nu_g = mu_g(T)/ rho_g(P, T)
        return nu_g

    def sigma(T):
        sigma = 0.06195* (1-(T/438.75))**1.277
        return sigma

    def Re_l(u, T, d):
        Re_l = u* d/ nu_l(T)
        return Re_l

    def Re_g(u, P, T, d):
        Re_g = u* d/ nu_g(P, T)
        return Re_g

    def h_l(u, P, T, d):
        Re_l_res = Re_l(u, T, d)
        Pr_l_res = Pr_l(T)
    
        if Re_l_res < 2300:
            Nu = 4.36
            h_l = Nu* k_l(T)/ d
    
        else:
            Nu = 0.023* Re_l_res**(0.8)* Pr_l_res**(0.4)
            h_l = Nu* k_l(T)/ d
        
        return h_l

    def h_g(u, P, T, d):
        Re_g_res = Re_g(u, P, T, d)
        Pr_g_res = Pr_g(T)
    
        if Re_g_res < 2300:
            Nu = 4.36
            h_g = Nu* k_g(T)/ d
        
        else:
            Nu = 0.023* Re_g_res**(0.8)* Pr_g_res**(0.4)
            h_g = Nu* k_g(T)/ d
        
        return h_g
    
    def tau_l(u, P, T, d):
        Re_l_res = Re_l(u, T, d)
        rho_l_res = rho_l(T)
    
        if 0 <= Re_l_res <= 0.01:
            tau_l = 0
        
        elif 0.01 < Re_l_res < 2300:
            f = 16/Re_l_res
            tau_l = 0.5* f* rho_l_res* u**2
        
        else:
            f = 0.0791* Re_l_res**(-0.25)
            tau_l = 0.5* f* rho_l_res* u**2
        
        return tau_l

    def tau_g(u, P, T, d):
        Re_g_res = Re_g(u, P, T, d)
        rho_g_res = rho_g(P, T)
    
        if 0 <= Re_g_res <=0.01:
            tau_g = 0
        
        elif 0.01 < Re_g_res < 2300:
            f = 16/Re_g_res
            tau_g = 0.5* f* rho_g_res* u**2
        
        else:
            f = 0.0791* Re_g_res**(-0.25)
            tau_g = 0.5* f* rho_g_res* u**2
        
        return tau_g
    
    def Delta_P_2p(u, T, d, x, m_dot, Delta_L):
        Re_g_sat = u*d*x*     prop(T,'rho_g',all_funcs)/prop(T,'mu_g',all_funcs)
        Re_l_sat = u*d*(1-x)* prop(T,'rho_l',all_funcs)/prop(T,'mu_l',all_funcs)
    
        def f(Re):
            if 0 <= Re <= 0.01:
                f = 0
            elif 0.01<Re<2300:
                f = 64/Re
            else:
                f = 0.3164*Re**(-0.25)
            
            return f
    
        def C(Re_g, Re_l):
            if Re_g<1500 and Re_l<1500:
                C = 5
            
            elif Re_g>=1500 and Re_l<1500:
                C = 12
            
            elif Re_g<1500 and Re_l>=1500:
                C = 10
            
            else:
                C = 20
            
            return C
    
        C_res = C(Re_g_sat, Re_l_sat)
    
        Delta_P_g = 8*f(Re_g_sat)*(m_dot**2 * x**2)*Delta_L/(prop(T,'rho_g', all_funcs)*math.pi**2 * d**5)
        Delta_P_l = 8*f(Re_l_sat)*(m_dot**2 * (1-x)**2)*Delta_L/(prop(T,'rho_l', all_funcs)*math.pi**2 * d**5)
    
        if Delta_P_g > 1e-7:
            X_LM = math.sqrt(Delta_P_l/Delta_P_g)
            Delta_P_2p = (1+ C_res* X_LM+ X_LM**2)* Delta_P_g
        
        else:
            X_LM = 0
            Delta_P_2p = Delta_P_g
    
        return Delta_P_2p, C_res, X_LM
    
    prop_dict = {
        "P_sat":P_sat,
        "rho_l":rho_l,
        "rho_g":rho_g,
        "Cp_l":Cp_l,
        "Cp_g":Cp_g,
        "lambda":lambda_lv,
        "k_l":k_l,
        "k_g":k_g,
        "mu_l":mu_l,
        "mu_g":mu_g,
        "Pr_l":Pr_l,
        "Pr_g":Pr_g,
        "nu_l":nu_l,
        "nu_g":nu_g,
        "sigma":sigma,
        "Re_l":Re_l,
        "Re_g":Re_g,
        "h_l":h_l,
        "h_g":h_g,
        "tau_l":tau_l,
        "tau_g":tau_g,
        "Delta_P_2p":Delta_P_2p
    }
    
    return prop_dict