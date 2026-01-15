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

design_dict = design_prop.design()
d = types.SimpleNamespace(**design_dict)
prop_dict = design_prop.prop(d.csv_path, d.csv_path_inv)
p = types.SimpleNamespace(**prop_dict)

T=350
P_sat=p.P_sat(T)
print(P_sat)

P,T, df_ec, M_dot, Qev, Qgr, Ploss = ec_flat.ec_flat(340, 330)
print(P,T,Qev,Qgr,Ploss)