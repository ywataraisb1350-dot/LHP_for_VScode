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

design_dict = design_prop.design()
d = types.SimpleNamespace(**design_dict)
prop_dict = design_prop.prop(d.csv_path, d.csv_path_inv)
p = types.SimpleNamespace(**prop_dict)



P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_gr, P_loss_wick_flat, P_loss_wick_gr, P_cap = ec_flat.ec_flat(340, 330)

rho = p.rho_g(P,T)
x, phase = 1, 'gas'
u = 4* M_dot/ (p.rho_g(P,T)* math.pi* d.d_i_vl **2)

u, P, T, rho, x, phase, df_vl, P_loss_vl, T_ave_vl, T_ini_vl = transline.trans_line(
    u, P, T, rho, x, phase, M_dot, d.L_vl, d.d_i_vl, d.d_o_vl, d.d_o_insu_vl, d.k_vl, d.k_insu_vl, d.h_out, d.T_amb, d.num_cal_vl)

u, P, T, rho, x, phase, df_cl, P_loss_cl, T_ave_cl, T_ini_cl = transline.trans_line(
    u, P, T, rho, x, phase, M_dot/2, d.L_cl, d.d_i_cl, d.d_o_cl, d.d_o_insu_cl, d.k_cl, d.k_insu_cl, d.h_sink, d.T_sink, d.num_cal_cl)

u, P, T, rho, x, phase, df_ll, P_loss_ll, T_ave_ll, T_ini_ll = transline.trans_line(
    u, P, T, rho, x, phase, M_dot, d.L_ll, d.d_i_ll, d.d_o_ll, d.d_o_insu_ll, d.k_ll, d.k_insu_ll, d.h_out, d.T_amb, d.num_cal_ll)
