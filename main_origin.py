import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math
#import sys
#import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

#設計用パラメータ入力用 長さは断らない限り[mm]

A_hs = 0.3072   #[m^2]
Q_load = 17000  #[W]

W_ec = 564  #[mm]
L_ec = 564  #[mm]
H_ec = 5    #[mm] =thickness
t_ec = 4    # = t_cc 肉厚
H_cc = 240  #[mm]
L_ec_cc = 7 #[mm]
k_ec = 16   #[W/m-K]

k_ec_flange = 398
k_cc_flange = 16
t_ec_flange = 4
t_cc_flange = 4

W_wick = 554
L_wick = 554
H_wick = 5          #=thickness mm
k_wick = 16         #[W/m-K]
r_max_pore = 5      #[micro m]
epsilon_wick = 0.6  #[-]
K_wick = 9e-13      #[m^2]
contact_angle = 10  #[deg]
n_gr = 55           #num of groove
w_gr = 5
h_gr = 5

d_i_vl = 40
d_o_vl = 42
L_vl = 15           #[m]
t_insu_vl = 50      #thickness mm
k_vl = 16           #[W/m-K]
k_insu_vl = 0.004   #[W/m-K]

d_i_cl = 40
d_o_cl = 42
L_cl = 25           #[m]
k_cl = 16
k_insu_cl = 1000    #実際はinsuなし，計算の便宜を図るため導入

d_i_ll = 30
d_o_ll = 32
L_ll = 15           #[m]
t_insu_ll = 50      #thickness mm
k_ll = 16           #[W/m-K]
k_insu_ll = 0.004   #[W/m-K]

T_amb = 30      #[do]
T_sink = 30
alpha = 10000   #[W/m^2-K] 蒸発熱伝達率A_ecベースの値
beta = 416     #[W/m^2-K] 凝縮熱伝達率
h_hs_ec = 4000  #[W/m^2-K] 熱源-蒸発器熱伝達率
h_out = 20.0    #[W/m^2-K] 決め打ち外部への放熱伝達率
h_sink = 500.0  #[W/m^2-K] 決め打ち外部へのコンデンサ放熱伝達率
grav_ac = 9.8   #gravity_acceralation

num_cal_ec, num_cal_vl, num_cal_cl, num_cal_ll = 100, 100, 200, 100
epsilon = 0.2
max_restarts = 100
iterations = 1000
learning_ratio = 2e-2
grad_clip_threshold = 50000
learning_rate_adam = 0.1 # 固定学習率より少し大きめに設定できることが多い
beta1 = 0.9
beta2 = 0.9
epsilon_adam = 1e-6
m_t = np.zeros(2) # モーメントベクトル
v_t = np.zeros(2)

T_ev=58.81886315519313+273.15
T_ec=63.93108468711017+273.15

#ここまで


#パラ変換
W_ec = W_ec*1e-3  #[m]
L_ec = L_ec*1e-3  #[m]
H_ec = H_ec*1e-3  #[m] =thickness
t_ec = t_ec*1e-3
H_cc = H_cc*1e-3  #[m]
L_ec_cc = L_ec_cc*1e-3 #[m]
A_ec = W_ec* L_ec   #menseki mo tsuideni

t_ec_flange = t_ec_flange*1e-3
t_cc_flange = t_cc_flange*1e-3

W_wick = W_wick*1e-3
L_wick = L_wick*1e-3
H_wick = H_wick*1e-3          #=thickness mm
A_wick = W_wick* L_wick #mensekimo tsuideni
r_max_pore = r_max_pore*1e-6      #[m]
contact_angle = math.radians(contact_angle)  #[rad]
w_gr = w_gr*1e-3
h_gr = h_gr*1e-3

A_ec = W_wick* L_wick* 1.5 # imadake!!!!!!!!!!
L_ec, W_ec = math.sqrt(A_ec), math.sqrt(A_ec)
G_ec_ccc = A_ec* 0.2/ (t_ec_flange/ k_ec_flange + t_cc_flange/ k_cc_flange)

d_e_ccside_1 = 1.3* (((W_ec- 2*t_ec)* H_cc)**0.625)/ (((W_ec- 2*t_ec)+ H_cc)**0.25)
d_e_ccside_2 = 1.3* (((L_ec- 2*t_ec)* H_cc)**0.625)/ (((L_ec- 2*t_ec)+ H_cc)**0.25)
d_e_ccside = (d_e_ccside_1+d_e_ccside_2)/2
d_gr = (4* w_gr* h_gr)/ (2*w_gr+ 2*h_gr)

d_i_vl = d_i_vl*1e-3
d_o_vl = d_o_vl*1e-3
t_insu_vl = t_insu_vl*1e-3
d_o_insu_vl = d_o_vl+ 2* t_insu_vl

d_i_cl = d_i_cl*1e-3
d_o_cl = d_o_cl*1e-3

d_i_ll = d_i_ll*1e-3
d_o_ll = d_o_ll*1e-3
t_insu_ll = t_insu_ll*1e-3
d_o_insu_ll = d_o_vl+ 2* t_insu_ll

T_amb = T_amb+ 273.15  #[k]
T_sink = T_sink+ 273.15
#変換ここまで

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

def prop(T, label, funcs_dict):
    if label not in funcs_dict:
        print(f"Error: Label '{label}' not found in the CSV data.")
        return None
    
    target_function = funcs_dict[label]
    prop_array = target_function(T)
    
    # .item() を使ってPythonの数値に変換してから返す
    prop_scalar = prop_array.item()
    
    return prop_scalar
    
    return prop

all_funcs = create_interpolation_functions('R1233_v2.csv') #prop関数の第三引数
sat_inv = create_interpolation_functions('R1233_inv.csv')

def P_sat(T):
    P_sat = prop(T, 'P_sat', all_funcs)
    return P_sat

def rho_l(T):
    rho_l = prop(T, 'rho_l', all_funcs)
    return rho_l

def rho_g(P,T):
    rho_g = prop(T, 'rho_g', all_funcs)
    #rho_g = P* 0.1305/(8.311* T)
    return rho_g

def Cp_l(T):
    Cp_l = prop(T, 'C_p_l', all_funcs)
    return Cp_l

def Cp_g(T):
    Cp_g = prop(T, 'C_p_g', all_funcs)
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
'''
def h_con_shah(M_dot, x, P, T, d_i):
'''
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

def G_gas(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex):
    h_in = h_g(u,P,T,d_i)
    #h_ex = (T-T_ex)*?
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_gas = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_gas, 1/R_1, 1/R_2, R_3, 1/R_4

def G_liq(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex):
    h_in = h_l(u, P, T, d_i)
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_liq = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_liq, 1/R_1, 1/R_2, R_3, 1/R_4

def G_mix(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex, M_dot, x):
    h_in = beta
    h_in = h_con(u, M_dot, x, P, T, d_i)
    R_1 = 1/ (h_in* math.pi* d_i* Delta_L)
    R_2 = math.log(d_o/d_i)/ (2* math.pi* k* Delta_L)
    R_3 = math.log(d_o_insu/d_o)/ (2* math.pi* k_insu* Delta_L)
    R_4 = 1/ (h_ex* math.pi* d_o_insu* Delta_L)
    G_mix = 1/ (R_1+ R_2+ R_3+ R_4)
    
    return G_mix, 1/R_1, 1/R_2, R_3, 1/R_4, h_in
        
def Delta_P_2p(u, d, T, x, M_dot, Delta_L):
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
    
    Delta_P_g = 8*f(Re_g_sat)*(M_dot**2 * x**2)*Delta_L/(prop(T,'rho_g', all_funcs)*math.pi**2 * d**5)
    Delta_P_l = 8*f(Re_l_sat)*(M_dot**2 * (1-x)**2)*Delta_L/(prop(T,'rho_l', all_funcs)*math.pi**2 * d**5)
    
    if Delta_P_g > 1e-7:
        X_LM = math.sqrt(Delta_P_l/Delta_P_g)
        Delta_P_2p = (1+ C_res* X_LM+ X_LM**2)* Delta_P_g
        
    else:
        X_LM = 0
        Delta_P_2p = Delta_P_g
    
    return Delta_P_2p, C_res, X_LM

def h_con(u, M_dot, x, P, T, d_i):
    """
    論文(Shah, 2016)の相関式に基づき、管内凝縮熱伝達率を計算する。
    "Comprehensive correlation #1" を実装。水平管を想定。

    Args:
        G (float): 質量流束 [kg/m^2·s]
        x (float): 乾き度 (Vapor quality) [-]
        P_sat (float): 飽和圧力 [Pa]
        T_sat (float): 飽和温度 [K]
        D (float): 管内径 [m]

    Returns:
        float: 二相熱伝達率 (h_TP) [W/m^2·K]
    """
    # 重力加速度 [m/s^2]
    
    # R1233zd(E) の臨界圧力を設定 [Pa]
    P_critical = 3.56e6 

    # 乾き度が0に近い、または1に近い場合は計算を避ける
    #if x < 0.001 or x > 0.99999:
        # この範囲では相関式の精度が落ちる可能性があるため、
        # 単相流の計算に切り替えるなどの処理が望ましい
        #return None 

    # --- 1. 物性値の取得 ---
    
    # --- 2. 無次元数の計算 ---
    # 換算圧力 Pr (論文中の P_r)
    Pr_reduced = P / P_critical
    G = M_dot/(math.pi* d_i**2* 0.25)

    # 液単相流と仮定した場合のレイノルズ数 Re_lo (Eq. 6の前)
    Re_lo = G* (1- x)* d_i/ mu_l(T)
    # Shahの相関パラメータ Z (Eq. 7)
    Z = ((1 / x) - 1)**0.8 * Pr_reduced**0.4

    # 全量が気相と仮定した場合のウェーバー数 We_GT (Eq. 13)
    We_GT = (G**2 * d_i) / (prop(T,'rho_g',all_funcs) * sigma(T))
    
    # 無次元蒸気速度 J_g (Eq. 10)
    Jg_denominator = math.sqrt(grav_ac * d_i * prop(T,'rho_g',all_funcs) * (rho_l(T) - prop(T,'rho_g',all_funcs)) )
    if Jg_denominator < 1e-9: return None # ゼロ除算防止
    J_g = (x * G) / Jg_denominator

    # --- 3. 熱伝達率の各要素を計算 ---
    # 液単相流と仮定した場合の熱伝達率 h_lo (Eq. 6)
    # Dittus-Boelter 式を使用
    h_lo = 0.023 * (Re_lo**0.8) * (Pr_l(T)**0.4) * k_l(T) / d_i
    
    # 対流凝縮が支配的な場合の熱伝達率 h_I (Eq. 1)
    # (論文では粘性比の項が追加されているが、Correlation #1では使わない)
    if Z < 1e-9: Z = 1e-9 # ゼロ除算防止
    h_I = h_lo* ( 1+ 1.128* x**0.817* (rho_l(T)/ rho_g(P, T))**0.3685 * (mu_l(T)/mu_g(T))**0.2363 * (1- mu_g(T)/mu_l(T))**2.144 * Pr_l(T)**(-1) )
    # 重力支配（層流膜状凝縮）の場合の熱伝達率 h_Nu (Eq. 2)
    # Nusseltの式を修正したもの
    if Re_lo < 1e-9: Re_lo = 1e-9 # ゼロ除算防止
    h_Nu_term1 = 1.32 * (Re_lo**(-1/3))
    h_Nu_term2 = ((rho_l(T) * (rho_l(T) - prop(T,'rho_g',all_funcs)) * grav_ac * (k_l(T)**3)) / (mu_l(T)**2))**(1/3)
    h_Nu = h_Nu_term1 * h_Nu_term2

    # --- 4. 熱伝達レジームの判定 (水平管の場合, Section 4.1) ---
    # Regime I の条件式 (Eq. 23)
    Jg_crit_I = 0.98 * (Z + 0.263)**(-0.62)
    
    # Regime III の条件式 (Eq. 24)
    Jg_crit_III = 0.95 * (1.254 + 2.27 * Z**1.249)**(-1)

    regime = "II" # デフォルトは Regime II
    if We_GT > 100 and J_g >= Jg_crit_I:
        regime = "I"
    elif We_GT > 20 and J_g <= Jg_crit_III:
        regime = "III"

    # --- 5. レジームに応じて最終的な熱伝達率 h_TP を計算 ---
    h_TP = 0.0
    if regime == "I":   # (Eq. 3)
        h_TP = h_I
    elif regime == "II":  # (Eq. 4)
        h_TP = h_I + h_Nu
    elif regime == "III": # (Eq. 5)
        h_TP = h_Nu
        
    return h_TP

def ec_flat(T_ec, T_ev):
    Q_ev = alpha* A_wick* (T_ec-T_ev)
    M_dot = Q_ev/ lambda_lv(T_ev)
    Delta_L = L_ec/ num_cal_ec
    u_max_times_rho = M_dot/(w_gr* h_gr* n_gr)
    
    Q_gr, P_loss_ec = 0, 0
    u, P, T = 0, prop(T_ev, 'P_sat', all_funcs), T_ev
    ec_result = []
    current_data = {
            'step': 'start',
            'distance':'start',
            'u': u,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho_g': rho_g(P,T),
            'm_dot':M_dot* (Delta_L)/ (n_gr* L_wick),
            'Re':Re_g(u, P, T, d_gr),
            'h_g':h_g(u, P, T, d_gr),
            'tau_g':tau_g(u, P, T, d_gr),
            'Q_gr':Q_gr,
            'Sigma_Plos_ec':P_loss_ec,
            'A_recieve_heat':Delta_L* w_gr+ 2* Delta_L* h_gr
            }
    ec_result.append(current_data)
    
    for i in range(num_cal_ec+ 1):
        
        T_next = ( (h_g(u, P, T, d_gr)*(Delta_L* w_gr+ 2* Delta_L* h_gr)*(T_ec- T_ev)
             + (M_dot*i*Delta_L* Cp_g(T)* T/ (n_gr* L_wick))+ (M_dot*Delta_L* Cp_g(T_ev)* T_ev/ (n_gr* L_wick)))
             / (  (M_dot*i*Delta_L* Cp_g(T)/ (n_gr* L_wick))+ (M_dot*Delta_L* Cp_g(T_ev)/ (n_gr* L_wick)) ))
        
        u_max = u_max_times_rho/ rho_g(P, T)
        P_next = P- 4* tau_g(u_max, P, T, d_gr)* Delta_L/ d_gr
        
        u = M_dot* (i+ 1)* Delta_L/(n_gr* L_wick* rho_g(P_next, T_next)* w_gr* h_gr)
        
        Q_gr = Q_gr+ (M_dot*i*Delta_L* Cp_g(T)* (T_next- T)/ (n_gr* L_wick))+ (M_dot*Delta_L* Cp_g(T_ev)* (T_next- T_ev)/ (n_gr* L_wick))* n_gr
        P_loss_ec = P_loss_ec+ P- P_next
        
        P, T = P_next, T_next
        current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'u_max':u_max,
            'P[kPa]': P*1e-3,
            'T': T-273.15,
            'rho_g': rho_g(P,T),
            'm_dot':M_dot* (i* Delta_L)/ (n_gr* L_wick),
            'Re':Re_g(u, P, T, d_gr),
            'mu_g':mu_g(T),
            'h_g':h_g(u, P, T, d_gr),
            'tau_g':tau_g(u, P, T, d_gr),
            'Q_gr':Q_gr,
            'Sigma_Plos_ec':P_loss_ec,
            'A_recieve_heat':Delta_L* w_gr+ 2* Delta_L* h_gr
            }
        ec_result.append(current_data)
        
    df_ec = pd.DataFrame(ec_result)
    
    return P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_ec

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
            T_next = T- G_gas_val* (T- T_ex)/ (M_dot* Cp_g(T))
            P_next = P- 4* tau_g(u, P, T, d_i)* Delta_L/ d_i
            P_loss_gas = P_loss_gas+ (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_gas = Q_release_gas+ G_gas_val* (T- T_ex)
            Delta_T = T-T_next
            P, T = P_next, T_next
            rho = rho_g(P, T)
            u = 4* M_dot/ (rho* math.pi* d_i**2)
                
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'P_sat[kPa]':P_sat(T)*1e-3,
            "P_loss_all_phase":P_loss_all_phase,
            'T': T-273.15,
            'rho': rho,
            'phase':phase,
            'x':x,
            'm_dot':u*rho*math.pi*0.25*d_i**2,
            'P_loss_gas':P_loss_gas,
            'Q_release_gas':Q_release_gas,
            'Delta_T':Delta_T,
            'G_in_gas':G_in,
            'G_pipe_gas':G_pipe,
            'R_insu_gas':R_insu,
            'G_ex_gas':G_ex
            }
            result.append(current_data)
            
            if P>= P_sat(T):
                #P = P_sat(T)
                x = 0.999999999
                phase = 'mix'
            
        elif phase=='mix':
            
            Delta_P_2p_val, C, X_LM = Delta_P_2p(u, d_i, T, x, M_dot, Delta_L)
            if X_LM<1e-3:
                X_LM = 1e-3
            G_gas_val, G_in, G_pipe, R_insu, G_ex = G_gas(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            G_mix_val, G_in, G_pipe, R_insu, G_ex,h_in = G_mix(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex, M_dot, x)
            G_liq_val, G_in, G_pipe, R_insu, G_ex = G_liq(u, P, T, Delta_L, d_i, d_o, d_o_insu, k, k_insu, h_ex, T_ex)
            P_next = P- Delta_P_2p_val
            T_next = prop(P_next, 'T', sat_inv)
            
            G = G_mix_val
            '''
            if 0.85 < x <=1.5:
                G = G_gas_val
            elif 0.15 <= x <= 0.85:
                G = G_mix_val
            elif x < 0.15:
                G = G_liq_val
            '''
            x = x+ (x* Cp_g(T)+ (1-x)* Cp_l(T))* (T- T_next)/ lambda_lv(T)- G* (T- T_ex)/ (M_dot* lambda_lv(T))
            if x>0.999999999:
                phase, x = 'gas', 1
                
            elif x <= 1e-6:
                phase, x = 'liq', 0
            u = max([ (M_dot* x/ (prop(T_next, 'rho_g', all_funcs)* math.pi* 0.25* d_i**2) ), ( M_dot/ (rho_l(T_next)* math.pi* 0.25* d_i**2) ) ])
            
            P_loss_mix = P_loss_mix + (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_mix = Q_release_mix + G_mix_val* (T- T_ex)
            Delta_T = T- T_next
            P, T = P_next, T_next
            rho = prop(T, 'rho_g', all_funcs)
            
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            'P_loss_all_phase':P_loss_all_phase,
            'T': T-273.15,
            'rho': '-',
            'phase':phase,
            'x':x,
            'm_dot':x*u*prop(T_next, 'rho_g', all_funcs)*math.pi*0.25*d_i**2,
            'P_loss_mix':P_loss_mix,
            'LM_C':C,
            'LM_X':X_LM,
            'H_con':h_in,
            'Delta_P_2p':Delta_P_2p_val,
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
            T_next = T- G_liq_val* (T- T_ex)/ (M_dot* Cp_l(T))
            P_next = P- 4* tau_l(u, P, T, d_i)* Delta_L/ d_i
            P_loss_liq = P_loss_liq+ (P- P_next)
            P_loss_all_phase = P_loss_all_phase+ (P- P_next)
            Q_release_liq = Q_release_liq+ G_liq_val* (T- T_ex)
            Delta_T = T-T_next
            P, T = P_next, T_next
            rho = rho_l(T)
            u = 4* M_dot/ (rho* math.pi* d_i**2)
                
            current_data = {
            'step': i,
            'distance':i* Delta_L,
            'u': u,
            'P[kPa]': P*1e-3,
            "P_loss_all_phase":P_loss_all_phase,
            'T': T-273.15,
            'rho': rho,
            'phase':phase,
            'x':x,
            'm_dot':u*rho*math.pi*0.25*d_i**2,
            'P_loss_liq':P_loss_liq,
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
    return u, P, T, rho, x, phase, df_line, P_loss, T_ave

def G_ccc_ccin(T):
    h_l_ccc = 4.36* k_l(T)/ d_e_ccside
    h_g_ccc = 4.36* k_g(T)/( 4*(W_ec- 2*t_ec)*(L_ec- 2*t_ec)/(2*(W_ec- 2*t_ec) + 2*(L_ec- 2*t_ec)) )
    G_ccc_ccin= h_l_ccc* ((W_ec- 2*t_ec)* H_cc* 2+(L_ec- 2*t_ec)* H_cc* 2) + h_g_ccc* (W_ec- 2*t_ec)*(L_ec- 2*t_ec)
    return G_ccc_ccin

def eval_func(T_ec, T_ev):
    P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_ec = ec_flat(T_ec,T_ev)
    rho = rho_g(P,T)
    x, phase = 1, 'gas'
    u = 4* M_dot/ (rho_g(P,T)* math.pi* d_i_vl**2)
    u, P, T, rho, x, phase, df_vl, P_loss_vl, T_ave_vl = trans_line(u,P,T,rho,x,phase,M_dot, L_vl, d_i_vl, d_o_vl, d_o_insu_vl, k_vl, k_insu_vl, h_out, T_amb, num_cal_vl)
    u, P, T, rho, x, phase, df_cl, P_loss_cl, T_ave_cl = trans_line(u,P,T,rho,x,phase,M_dot, L_cl, d_i_cl, d_o_cl, d_o_cl, k_cl, 1000, h_sink, T_sink, num_cal_cl)
    u, P, T, rho, x, phase, df_ll, P_loss_ll, T_ave_ll = trans_line(u,P,T,rho,x,phase,M_dot, L_ll, d_i_ll, d_o_ll, d_o_insu_ll, k_ll, k_insu_ll, h_out, T_amb, num_cal_ll)
    
    T_ccin = prop(P, 'T', sat_inv)
    T_ccc =( (G_ccc_ccin(T_ccin)* T_ccin + h_out* (A_ec+ (2*W_ec + 2*L_ec)* H_cc)* T_amb + G_ec_ccc* T_ec)/
        (G_ccc_ccin(T_ccin) + h_out* (A_ec+ (2*W_ec + 2*L_ec)* H_cc) + G_ec_ccc) )
    
    k_eff = epsilon_wick* k_l(T_ccin)+ (1- epsilon_wick)* k_wick
    Q_ec_wick_ccin = 3*k_eff* A_wick* (T_ev- T_ccin)/ H_wick
    Q_ec_ccc = G_ec_ccc* (T_ec- T_ccc)
    Q_ec_amb = h_out* (A_ec- A_hs + (H_ec+ h_gr+ H_wick)* W_ec* 2 + (H_ec+ h_gr+ H_wick)* L_ec* 2)* (T_ec- T_amb)
    T_hs = (Q_load + h_out* A_hs* T_amb + h_hs_ec* A_ec* T_ec)/(h_out* A_hs + h_hs_ec* A_ec)
    Q_hs_amb = h_out* A_hs* (T_hs- T_amb)
    Q_ec_in = Q_load#- Q_hs_amb
    Q_ec_out = Q_ev+ Q_gr+ Q_ec_ccc+ Q_ec_wick_ccin+ Q_ec_amb
    eval_ec = (100*(Q_ec_in- Q_ec_out)/ Q_load)**2
    Q_cc_ll = M_dot* Cp_l(T)* (T_ccin- T)
    Q_ccc_ccin = G_ccc_ccin(T_ccin)* (T_ccc-T_ccin)
    eval_ccin = (100*(Q_ccc_ccin+ Q_ec_wick_ccin- Q_cc_ll)/ (0.1* Q_load))**2
    
    Q_ccc_amb = h_out* (A_ec+ (2*W_ec + 2*L_ec)* H_cc)* (T_ccc- T_amb)
    
    return (
        eval_ec, eval_ccin, df_ec, P_loss_ec, df_vl, P_loss_vl, T_ave_vl, df_cl, P_loss_cl, T_ave_cl, df_ll, P_loss_ll, T_ave_ll,
        k_eff, Q_ec_ccc, Q_ec_wick_ccin, Q_ec_amb, T_hs, Q_hs_amb, Q_ec_in, Q_ec_out, Q_cc_ll, Q_ccc_ccin, Q_ccc_amb, T, T_ccin, T_ccc, P, Q_ev, Q_gr)

(
eval_ec, eval_ccin, df_ec, P_loss_ec, df_vl, P_loss_vl, T_ave_vl, df_cl, P_loss_cl, T_ave_cl, df_ll, P_loss_ll, T_ave_ll,
k_eff, Q_ec_ccc, Q_ec_wick_ccin, Q_ec_amb, T_hs, Q_hs_amb, Q_ec_in, Q_ec_out, Q_cc_ll, Q_ccc_ccin, Q_ccc_amb, T, T_ccin, T_ccc, P, Q_ev, Q_gr
)= eval_func(T_ec, T_ev)


def design_and_result():
    des_and_res={
        "T_ec":T_ec-273.15,
        "T_ev":T_ev-273.15,
        "T_ave_vl":T_ave_vl-273.15,
        "T_ave_cl":T_ave_cl-273.15,
        "T_ave_ll":T_ave_ll-273.15,
        "T_ll_out":T-273.15,
        "T_ccin":T_ccin-273.15,
        "T_ccc":T_ccc-273.15,
        
        "P_ccin":P_sat(T_ccin),
        "P_initial=P_ev":P_sat(T_ev),
        "P_ll_out":P,
        
        "G_ccc_ccin":G_ccc_ccin(T_ccin),
        "Q_ec_wick_ccin":Q_ec_wick_ccin,
        "Q_ec_ccc":Q_ec_ccc,
        "Q_ec_amb":Q_ec_amb,
        "T_hs":T_hs-273.15,
        #"Q_hs_amb":Q_hs_amb,
        "Q_load-hs_amb":Q_ec_in,
        "Q_ev":Q_ev,
        "Q_gr":Q_gr,
        "Q_ev+gr+ccc+wick+amb":Q_ec_out,
        "eval_1":eval_ec,
   
        "Q_cc_ll":Q_cc_ll,
        "Q_ccc_ccin":Q_ccc_ccin,
        "Q_ccc_amb":Q_ccc_amb,
        "eval_2":eval_ccin,
        
        "P_loss_ec":P_loss_ec,
        "P_loss_vl":P_loss_vl,
        "p_loss_cl":P_loss_cl,
        "P_loss_ll":P_loss_ll,
        "Ploss_kasan":P_loss_ec+ P_loss_vl+ P_loss_cl+ P_loss_ll,
        "Ploss_gyakusan":P_sat(T_ev)-P,
        
        "compere":"my-W.sennsei",
        
        "Q_load_com":Q_load-17000,
        "Q_ev_com":Q_ev- 15567.551,
        "Q_ec_ccc_com":Q_ec_ccc- 671.768,
        "Q_gr_com":Q_gr-263.795,
        "Q_ec_wick_ccin_com":Q_ec_wick_ccin- 371.473,
        "Q_ec_amb_com":Q_ec_amb- 125.412,
        "Q_ccc_ccin_com":Q_ccc_ccin- (-1.406),
        "Q_sub_com":Q_cc_ll- 342.961,
        "Q_ccc_amb_com":Q_ccc_amb- 670.362,
        "Ploss_ec":P_loss_ec-383,
        "Ploss_vl":P_loss_vl- 647,
        "Ploss_cl":P_loss_cl- 1265,
        "Ploss_ll":P_loss_ll- 104,
        
        "A_hs":A_hs,
        "Q_load":Q_load,

        "W_ec":W_ec,  
        "L_ec":L_ec, 
        "H_ec":H_ec,  
        "H_cc":H_cc, 
        "L_ec_cc":L_ec_cc,
        "k_ec":k_ec,  

        "W_wick":W_wick,
        "L_wick":L_wick,
        "H_wick":H_wick,      
        "k_wick":k_wick,     
        "r_max_pore":r_max_pore,     
        "epsilon_wick":epsilon_wick, 
        "k_wick":K_wick,     
        "contact_ang":contact_angle, 
        "n_gr":n_gr,      
        "w_gr":w_gr,
        "h_gr":h_gr,
        "K_eff":k_eff,

        "d_i_vl":d_i_vl,
        "d_o_vl":d_o_vl,
        "L_vl":L_vl,       
        "t_insu_vl":t_insu_vl,    
        "k_vl":k_vl,
        "k_insu_vl":k_insu_vl,

        "d_i_cl":d_i_cl,
        "d_o_cl":d_o_cl,
        "L_cl":L_cl,
        "k_cl":k_cl,
        "k_insu_cl":k_insu_cl,

        "d_i_ll":d_i_ll,
        "d_o_ll":d_o_ll,
        "L_ll":L_ll,
        "t_insu_ll":t_insu_ll,
        "k_ll":k_ll,
        "k_insu_ll":k_insu_ll,

        "T_amb":T_amb,
        "T_sink":T_sink,
        "alpha":alpha,
        "beta":beta,
        "h_hs_ec":h_hs_ec,
        "h_out":h_out,
        "h_sink":h_sink,
        "grav_ac":grav_ac,

        "num_cal_ec":num_cal_ec, 
        "num_cal_vl":num_cal_vl,
        "num_cal_cl":num_cal_cl,
        "num_cal_ll":num_cal_ll,
        "epsilon":epsilon,
        "max_restart":max_restarts,
        "iterations":iterations,
        "learning_raito":learning_ratio,
        "grad_clip":grad_clip_threshold,
        "learn_rate_adam":learning_rate_adam,
        "beta1":beta1,
        "beta2":beta2,
        "epsilon_adam":epsilon_adam
    }
    return des_and_res

design_and_result_fordf = design_and_result()
df_res_and_para = pd.DataFrame(design_and_result_fordf.items(), columns=['lavel', 'val'])
now= datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(timestamp, exist_ok=True)
file_path_ec = os.path.join(timestamp, f'ec_{timestamp}.csv')
df_ec.to_csv(file_path_ec, index=False)
file_path_vl = os.path.join(timestamp, f'vl_{timestamp}.csv')
df_vl.to_csv(file_path_vl, index=False)
file_path_con = os.path.join(timestamp, f'cl_{timestamp}.csv')
df_cl.to_csv(file_path_con, index=False)
file_path_ll = os.path.join(timestamp, f'll_{timestamp}.csv')
df_ll.to_csv(file_path_ll, index=False)
file_path_des = os.path.join(timestamp, f'result_and_parameter_{timestamp}.csv')
df_res_and_para.to_csv(file_path_des, index=False)

P_loss_labels = ["my_ec", "N.W._ec", "my_vl", "N.W._vl", "my_cl",  "N.W._cl", "my_ll", "N.W._ll"]
P_loss_values = [P_loss_ec, 383, P_loss_vl, 647, P_loss_cl, 1265, P_loss_ll, 104]
plt.figure(figsize=(10, 6))
bars = plt.bar(P_loss_labels, P_loss_values, color=['skyblue', 'lightgreen', 'salmon', 'plum'])
plt.title(f'Tec={T_ec-273.15} tev={T_ev-273.15}', fontsize=16)
plt.ylabel('P_loss [Pa]', fontsize=12)
plt.xticks(fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
graph_file_path = os.path.join(timestamp, f'pressure_loss_{timestamp}.png')
plt.savefig(graph_file_path)

Q_labels = ['myQ_ec_ccc','NW Q_ec_ccc',
            'myQ_gr','NW Q_gr',
            'myQ_wick_ccin','NW Q_wick_ccin',
            'myQ_ec_amb','NW Q_ec_amb',
            'myQ_ccc_ccin','NW Q_ccc_ccin',
            'myQ_sub','NW Q_sub',
            'myQ_ccc_amb','NW Q_ccc_amb']
Q_values = [Q_ec_ccc, 671.768,
            Q_gr, 263.795,
            Q_ec_wick_ccin, 371.473,
            Q_ec_amb, 125.412,
            Q_ccc_ccin, -1.406,
            Q_cc_ll, 342.961,
            Q_ccc_amb, 670.362]
plt.figure(figsize=(20, 10)) 
bars = plt.bar(Q_labels, Q_values, color='c')
plt.title(f'Tec={T_ec-273.15} tev={T_ev-273.15}', fontsize=16)
plt.ylabel('[W]', fontsize=12)
plt.xticks(rotation=15, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
heat_balance_graph_path = os.path.join(timestamp, f'Q_balance_{timestamp}.png')
plt.savefig(heat_balance_graph_path)

T_labels = [#'myT_hs','NW T_hs',
            'myT_ec','NW T_ec',
            'myT_ev','NW T_ev',
            'myT_ccin','NW T_ccin',
            'myT_vl_ave','NW T_vl_ave',
            'myT_cl_ave','NW T_cl_ave',
            'myT_ll_ave','NW T_ll_ave']
T_values = [#T_hs-273.15, 77,
            T_ec-273.15, 63.1,
            T_ev-273.15, 58.1,
            T_ccin-273.15, 57.9,
            T_ave_vl-273.15, 60.7,
            T_ave_cl-273.15, 57.9,
            T_ave_ll-273.15, 55.2]
plt.figure(figsize=(20, 10)) 
bars = plt.bar(T_labels, T_values, color='c')
plt.title(f'Tec={T_ec-273.15} Tev={T_ev-273.15}', fontsize=16)
plt.ylabel('[W]', fontsize=12)
plt.xticks(rotation=15, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
temp_graph_path = os.path.join(timestamp, f'temp_{timestamp}.png')
plt.savefig(temp_graph_path)

print('T_ec=', T_ec-273, 'T_ev=', T_ev-273)
print('eval_ec=', eval_ec, 'eval_ccin=', eval_ccin)