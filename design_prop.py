import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime

def design():
    #design parameter input zone
    W_ec = 320  #mm
    L_ec = 320  #mm
    t_ec_bt = 5 #mm
    t_ec = 10   #mm
    t_ec_up = 5 #mm
    k_ec = 390 #conductivity
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
    t_kubire = 8
    w_kubire = 25
    l_kubire = 150

    W_wick = 280
    L_wick = 280
    H_wick = 8          #=thickness mm
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
    alpha = 10000   #[W/m^2-K] 蒸発熱伝達率A_ecベースの値
    #beta = 416     #[W/m^2-K] 凝縮熱伝達率
    h_hs_ec = 4000  #[W/m^2-K] 熱源-蒸発器熱伝達率
    h_out = 20.0    #[W/m^2-K] 決め打ち外部への放熱伝達率
    h_sink = 800.0  #[W/m^2-K] 決め打ち外部へのコンデンサ放熱伝達率
    grav_ac = 9.8   #gravity_acceralation
    csv_path = 'R1233zdE.csv'
    csv_path_inv = 'R1233zdE_inv.csv'

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
    
    prop_dict = {
        "P_sat":P_sat,
        "rho_l":rho_l
    }
    
    return prop_dict