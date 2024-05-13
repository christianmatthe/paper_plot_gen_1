########## imports and setup
import numpy as np
import matplotlib.pyplot as plt
import os
import time
#from datetime import datetime, tzinfo
import datetime as dt
from scipy.interpolate import interp1d
import dill
#from scipy.stats import sigmaclip
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.special import binom
from numpy.linalg import inv
import json


cwd = os.getcwd()
print("cwd:", os.getcwd())


from wire_analysis.flow_on_off_cycle_analysis import (load_dict,
                                                       make_result_dict)

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

plot_dir = (cwd + os.sep 
            + "output/")
data_dir = cwd + os.sep + ".." + os.sep + "data/"
os.makedirs(plot_dir, exist_ok=True)
#######################

if __name__ =="__main__": 
# ########################################
########## 2023-04-24  2023-04-21_1sccm_15A_TC_z-scan_jf_wire
    run_name = "2023-04-21_1sccm_15A_TC_z-scan_jf_wire_recalib"
    # data_name = "2023-04-21_1sccm_15A_TC_z-scan_jf_wire"



    ext_dict = load_dict(data_dir + run_name + os.sep 
                                + "ext_dict")
    result_dict = make_result_dict(ext_dict)
    print(result_dict)
    print(result_dict.items())

    # Plot directly from saved ext object:
    # Chose 18th position = z = 1.5mm  for paper
    for i in range(37):
        ext = ext_dict[i]["extractor"]
        z = ext_dict[i]["z"]
        plot_path = (plot_dir + run_name + os.sep 
                     + f"{i}_{z:.2f}mm" + os.sep
                     + "paper" )
        os.makedirs(plot_path)
        ext.plot_all_ABA_fit_paper(
                            plot_path = plot_path,
                                )

