import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime

#design parameter input zone
W_ec = 320  #mm
L_ec = 320  #mm
t_ec_bt = 5 #mm
t_ec = 10   #mm
t_ec_up = 5 #mm
k_ec = 16.3 #conductivity
H_ec = 40   #mm

r_cc = 160
t_cc_bt = 4
t_cc_up = 4
t_cc = 4
H_cc = 200

