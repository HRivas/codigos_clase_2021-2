# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:24:51 2021

@author: Ricardo
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

class backtest:
    
    def __init__(self):
        self.ric_long = 'TOTF.PA' # Numerator
        self.ric_short = 'REP.MC' # Denominator
        self.rolling_days = 20 # N
        self.level_1 = 1. # a
        self.level_2 = 2. # b
        self.data_cut = 0.7 # 70 % in-sample and 30 % out-of-sample
        self.data_type = 'in-sample' # in-sample out-of-sample
        self.data_table = None
        
    def load_timeseries(self):
        self.data_table = file_functions.load_synchronised_timeseries(ric_x=self.ric_long, ric_y=self.ric_short)