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

T=350
P_sat=p.P_sat(T)
print(P_sat)

P, T, df_ec, M_dot, Q_ev, Q_gr, P_loss_gr, P_loss_wick_flat, P_loss_wick_gr, P_cap = ec_flat.ec_flat(340, 330)

rho = p.rho_g(P,T)
x, phase = 1, 'gas'
u = 4* M_dot/ (p.rho_g(P,T)* math.pi* d.d_i_vl**2)

u, P, T, rho, x, phase, df_line, P_loss, T_ave, T_ini = transline.trans_line(
    u, P, T, rho, x, phase, M_dot, d.L_vl, d.d_i_vl, d.d_o_vl, d.d_o_insu_vl, d.k_vl, d.k_insu_vl, d.h_ex, d.T_ex, d.num_cal_vl)

print(u,P,T,x,'phase')