########## imports and setup
#from this import d
#from tracemalloc import start
#from audioop import avg
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from scipy.interpolate import interp1d
import dill
#from scipy.stats import sigmaclip
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from decimal import Decimal

# import functions from other file
from wire_analysis.Voltage_base_analysis import (prep_data_calib,
                                                 load_data)


#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

cwd = os.getcwd()
print("cwd:", os.getcwd())

plot_dir = (cwd + os.sep 
            + "output/")
data_dir = cwd + os.sep + ".." + os.sep + "data/"
os.makedirs(plot_dir, exist_ok=True)
#######################


def make_index_dict(data_dict):
    """
    create a dictionary that specifies the index range for which a certain
    current was applied
    """
    data = data_dict
    if data_dict["i_set_str"] is not None:
        # for new "SlowDash" data
        unique_i_arr = np.unique(data["i_set_str"])
        #print("unique_i_arr: ", unique_i_arr)

        index_dict={}
        for current in unique_i_arr:
            index_dict[f"{current}"] = []
            # find all datapoints with a certain current
            index_section = np.where(data["i_set_str"] == current)[0]
            if len(index_section) < 8:
                del index_dict[f"{current}"]
                continue
            # cut index section at non consectutive indexes
            # create list of indexes where previous index is not consecutive
            non_consec = [i for i in range(1,len(index_section))
            if (index_section[i]
                != index_section[i-1] + 1)
            ]
            # prepend 0th instance
            non_consec = [0] + non_consec

            # print(non_consec)
            index_dict[f"{current}"] = [
                index_section[non_consec[i]:non_consec[i+1]]
                if i != len(non_consec) - 1
                else index_section[non_consec[i]:]
                for i in range(len(non_consec))
                ]

    else:
        unique_i_arr = np.unique(data["i_set"])
        #print("unique_i_arr: ", unique_i_arr)

        index_dict={}
        for current in unique_i_arr:
            index_dict[f"{current}"] = []
            # find all datapoints with a certain current
            index_section = np.where(data["i_set"] == current)[0]
            if len(index_section) < 5:
                del index_dict[f"{current}"]
                continue
            # cut index section at non consectutive indexes
            # create list of indexes where previous index is not consecutive
            non_consec = [i for i in range(1,len(index_section))
            if (index_section[i]
                != index_section[i-1] + 1)
            ]
            # prepend 0th instance
            non_consec = [0] + non_consec

            # print(non_consec)
            index_dict[f"{current}"] = [
                index_section[non_consec[i]:non_consec[i+1]]
                if i != len(non_consec) - 1
                else index_section[non_consec[i]:]
                for i in range(len(non_consec))
                ]
    return index_dict 

def make_index_dict_sd(data_dict):
    """
    create a dictionary that specifies the index range for which a certain
    current was applied
    """
    data = data_dict
    unique_i_arr = np.unique(data["i_set_str"])
    #print("unique_i_arr: ", unique_i_arr)

    index_dict={}
    for current in unique_i_arr:
        index_dict[f"{current}"] = []
        # find all datapoints with a certain current
        index_section = np.where(data["i_set_str"] == current)[0]
        if len(index_section) < 5:
            del index_dict[f"{current}"]
            continue
        # cut index section at non consectutive indexes
        # create list of indexes where previous index is not consecutive
        non_consec = [i for i in range(1,len(index_section))
        if (index_section[i]
             != index_section[i-1] + 1)
        ]
        # prepend 0th instance
        non_consec = [0] + non_consec

        # print(non_consec)
        index_dict[f"{current}"] = [
            index_section[non_consec[i]:non_consec[i+1]]
            if i != len(non_consec) - 1
            else index_section[non_consec[i]:]
            for i in range(len(non_consec))
            ]
    return index_dict 

def plot_calib(data_dict, plotname,index_arr = None):
    if index_arr is None:
        v_series = data_dict["voltage"]
        dates = data_dict["dates"]
    else:
        v_series = np.array(data_dict["voltage"])[index_arr]
        dates = np.array(data_dict["dates"])[index_arr]
    
    sigma = np.std(v_series[4:])

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(dates,v_series*1000,".",# markersize=1,
            label = f"data, std = {sigma * 1000:.1e}")
    # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
    #         label = f"moving average {mavg_len}s")

    plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"Voltage [mV]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_dates"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

def plot_calib_sectioned(data_dict, plotname, clip_leading = 4):
    index_dict = make_index_dict(data_dict)

    # Plot each section individually
    for key in index_dict.keys():
        for i,section in enumerate(index_dict[key]):
            v_series = np.array(data_dict["voltage"])[section][clip_leading:]
            dates = np.array(data_dict["dates"])[section][clip_leading:]

            sigma = np.std(v_series)

            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()
            ax1.plot(dates,v_series*1000,".",# markersize=1,
                    label = f"data, std = {sigma * 1000:.1e}")
            # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
            #         label = f"moving average {mavg_len}s")

    plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"Voltage [mV]")

    plt.grid(True)
    # plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_sec"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

def plot_calib_section(data_dict, plotname, key_list, clip_leading = 4):
    index_dict = make_index_dict(data_dict)
    # index_dict = {key:index_dict[key] for key in key_list}
    # Plot each section individually
    for key in index_dict.keys():
        for i,section in enumerate(index_dict[key]):
            v_series = np.array(data_dict["voltage"])[section][clip_leading:]
            dates = np.array(data_dict["dates"])[section][clip_leading:]

            sigma = np.std(v_series)

            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()
            ax1.plot(dates,v_series*1000,".",# markersize=1,
                    label = f"{key}_{i}, std = {sigma * 1000:.1e}")
            # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
            #         label = f"moving average {mavg_len}s")

    plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"Voltage [mV]")

    plt.grid(True)
    #plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "keys" + f"{key_list}"+"_sec"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

def plot_calib_kappa(data_dict, plotname,index_arr = None):
    if index_arr is None:
        v_series = data_dict["voltage"]
        dates = data_dict["dates"]
    else:
        v_series = np.array(data_dict["voltage"])[index_arr]
        dates = np.array(data_dict["dates"])[index_arr]
    
    sigma = np.std(v_series[4:])

    v_series_masked = sigma_clip(v_series,masked = True)
    mask = np.logical_not(v_series_masked.mask)

    sigma = np.std(v_series[mask])

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(dates[mask],v_series[mask]*1000,".",# markersize=1,
            label = f"data, std = {sigma * 1000:.1e}")
    # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
    #         label = f"moving average {mavg_len}s")

    plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Time")
    ax1.set_ylabel(r"Voltage [mV]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_dates"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()


def R_err(i,i_err,v,v_err): 
    res = np.sqrt(
            ((1/i) * v_err )**2
            + ((v/(i**2)) * i_err)**2
                  )
    return res

def P_err(i,i_err,v,v_err): 
    res = np.sqrt(
            (i * v_err )**2
            + (v * i_err)**2
                  )
    return res

def raw_dict_to_avg(data_dict,  key_list = "all",
                    clip_leading = 4, clip_trailing =2, index_dict = "default"):
    if index_dict == "default":
        index_dict = make_index_dict(data_dict)
    if key_list == "all":
        key_list = index_dict.keys()

    #initialize dict 
    avg_dict = {
        "v" : [],
        "i" : [],
        "i_set" : [],
        "v_err" : [],
        "i_err" : [],
        "v_point_err" : [],
        "i_point_err" : [],
        "R_Pt_1000" : [],
        "R_Pt_1000_err" : [],
        "R_Pt_1000_point_err" : [],

    }
    # Add average date?

    # Split series and average each
    for key in key_list:
        for i,section in enumerate(index_dict[key]):
            v_series = np.array(data_dict["voltage"])[section][clip_leading:
                                                        -clip_trailing]
            i_series = np.array(data_dict["i_series"])[section][clip_leading:
                                                        -clip_trailing]
            i_set = np.array(data_dict["i_set"])[section][clip_leading:
                                                        -clip_trailing]
            dates = np.array(data_dict["dates"])[section][clip_leading:
                                                        -clip_trailing]
            R_Pt = np.array(data_dict["R_Pt_1000"])[section][clip_leading:
                                                        -clip_trailing]
            

            # convert unit: we want mA and mV
            v_series = v_series * 1000
            i_series = i_series * 1000

            n_meas = len(v_series)
            v_mean = np.average(v_series)
            i_mean = np.average(i_series)
            R_Pt_mean = np.average(R_Pt)
            i_set_mean = np.average(i_set)

            v_sigma = np.std(v_series)
            i_sigma = np.std(i_series)
            R_Pt_sigma = np.std(R_Pt)
            v_mean_err = v_sigma / np.sqrt(n_meas)
            i_mean_err = i_sigma / np.sqrt(n_meas)
            R_Pt_mean_err = R_Pt_sigma / np.sqrt(n_meas)

            avg_dict["v"].append(v_mean)
            avg_dict["i"].append(i_mean)
            avg_dict["i_set"].append(i_set_mean)
            avg_dict["v_point_err"].append(v_sigma) # stat error of 1 meas.
            avg_dict["i_point_err"].append(i_sigma)
            avg_dict["v_err"].append(v_mean_err) # error of the mean
            avg_dict["i_err"].append(i_mean_err)
            avg_dict["R_Pt_1000"].append(R_Pt_mean)
            avg_dict["R_Pt_1000_point_err"].append(R_Pt_sigma)
            avg_dict["R_Pt_1000_err"].append(R_Pt_mean_err)


    # convert to nummpy  arrays
    for key in avg_dict:
        avg_dict[key] = np.array(avg_dict[key])
    
    #make calculated quantities
    avg_dict["R"] = avg_dict["v"] / avg_dict["i"]
    avg_dict["P"] = avg_dict["v"] * avg_dict["i"]
    avg_dict["R_err"] = R_err(avg_dict["i"], avg_dict["i_err"],
                              avg_dict["v"], avg_dict["v_err"]
                                )
    avg_dict["P_err"] = P_err(avg_dict["i"], avg_dict["i_err"],
                              avg_dict["v"], avg_dict["v_err"]
                                )

    return avg_dict



def plot_R_vs_P(avg_dict, plotname):
    ############## Resistance vs power plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.errorbar(avg_dict["P"],
             avg_dict["R"],
             yerr = avg_dict["R_err"],
             xerr = avg_dict["P_err"],
             fmt = ".",
             # markersize=1,
             label = f"data")
    # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
    #         label = f"moving average {mavg_len}s")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

def fit_base_R(avg_dict, plotname):
    # possibly undercuts 2nd order correction effects (R is left uncorrected)
    # def R_func(i, m, R_0, i_offset):
    #     u = (R_0 * (i + i_offset))/(1 - m * (i + i_offset)**2)
    #     R = m* u*(i + i_offset) + R_0
    #     return R 
    def v_func(i, m, R_0, i_offset):
        v = (R_0 * (i + i_offset))/(1 - m * (i + i_offset)**2)
        return v

    # try a different idea
    # def R_func(i, m, R_0, i_offset):
    #     R = R_0 /(1 - m *(i+i_offset)**2)
    #     return R 

    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]
    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    #fit
    # print("fit ydata:", i_arr)
    # print('len(ydata):', len(i_arr))
    # print('len(v_err):', len(v_err))
    # print('len(avg_dict["v_err"]):', len(avg_dict["v_err"]))
    # print("np.isnan(avg_dict['v_err'])",np.isnan(avg_dict["v_err"]))
    # print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    popt, pcov = curve_fit(v_func, i_arr, v_arr, p0 = [0.135,66.14,-0.000105],
                           sigma = v_err, absolute_sigma=True,
                           #bounds = ((0.1,60,-0.00035),(10,75, -0.00025))
                           bounds = ((0.1,60,-0.0005),(10,75, 0.0005))
    )

    offset = popt[2]
    offset_err = pcov[2][2]
    i_arr_off = i_arr + offset
    i_err_off = np.sqrt(i_err **2 + offset_err **2)

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # old no errorbars
    # ax1.plot(i_arr_off * v_arr ,
    #          v_arr / i_arr_off,
    #          ".",# markersize=1,
    #          label = f"data, offset_{1000*offset:3f} [µA]")
    # # with errorbars
    ax1.errorbar(i_arr_off * v_arr ,
             v_arr / i_arr_off,
             yerr = R_err(i_arr_off, i_err_off, v_arr, v_err),
             xerr = P_err(i_arr_off, i_err_off, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data, offset_{1000*offset:3f} [µA]")

    # Plot Fit
    xdata=p_arr
    plt.plot(xdata, v_func(i_arr, *popt)/i_arr_off, 'r-',
         label=(('fit: m=%5.3f [Ohm/µW ], R_0=%5.3f [Ohm],' 
                  + '\n i_off=%5.6f [mA]')
                % tuple(popt))
                )
    # ax1.plot(t_avg_series-t_series[0],v_avg_series*1000,
    #         label = f"moving average {mavg_len}s")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # troubleshoot plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # ax1.plot(i_arr ,
    #          v_arr / i_arr,
    #          ".",# markersize=1,
    #          label = f"data, offset_{1000*offset:3f} [µA]")
    ax1.errorbar(i_arr ,
             v_arr / i_arr,
             yerr = avg_dict["R_err"][~np.isnan(avg_dict["i"])],
             xerr = avg_dict["i_err"][~np.isnan(avg_dict["i"])],
             fmt = ".",# markersize=1,
             label = f"data, offset_{1000*offset:3f} [µA]")

    # Plot Fit
    xdata=i_arr
    plt.plot(xdata, v_func(i_arr, *popt)/i_arr, 'r-',
         label=(('fit: m=%5.3f [Ohm/µW ], R_0=%5.3f [Ohm],' 
                  + '\n i_off=%5.6f [mA]')
                % tuple(popt))
                )
    ax1.set_xlabel(r"current [mA]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "trouble"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    return popt, pcov

def basic_R_over_P_calib(avg_dict, plotname):
    # fit 0.9mA to 1.1mA range directly with a
    def fit_func(P, m, R_0):
        R = R_0 + m * P
        return R


    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]
    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)
    # #fit
    popt, pcov = curve_fit(fit_func, p_arr, r_arr, p0 = [0.135,66.7],
                           sigma = R_err(i_arr, i_err, v_arr, v_err),
                           absolute_sigma=True,
                           bounds = ((0.05,60),(1.0,75))
    )

    #########Plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    plt.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [Ohm/µW ],
         R_0={:5.3f} $\pm$ {:2.1e} [Ohm]'''.format(
             popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
                  ))
                )

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #################### plot with residuals
    residuals = r_arr - fit_func(p_arr, *popt)
    chi2 = np.sum((residuals/r_err)**2)
    dof = len(residuals) - len(popt)
    chi2_red = chi2/dof

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    gs.update(#wspace=0.05
            hspace = 0.005
        )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ### ax1
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = r"data, $\chi^2_{red}$"+" = {:2.2f} ".format(chi2_red))

    # Plot Fit
    xdata=p_arr
    ax1.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [$\Omega$/µW],
         R$_0$={:5.3f} $\pm$ {:2.1e} [$\Omega$]'''.format(
             popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
                  ))
                )

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    ax1.grid(True)
    ax1.legend(shadow=True, fontsize = 13)
    # ax1.tight_layout()

    #ax 2

    ax2.errorbar(p_arr ,
             residuals,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    ax2.plot(xdata, 0.0 * fit_func(p_arr, *popt), 'r-',
        #  label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [Ohm/µW ],
        #  R_0={:5.3f} $\pm$ {:2.1e} [Ohm]'''.format(
        #      popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
        #           ))
                )
    ax2.set_xlabel(r"power [µW]")
    ax2.set_ylabel(r"Residuals [$\Omega$]")

    ax2.grid(True)

    #make custom pruning of uppper tick (do not plot ticks in upper 10%)
    #so that ax2 tick does nto interfere with  ax1 tick
    ax2.locator_params(axis="y", min_n_ticks = 3
                       )
    y_loc = ax2.yaxis.get_majorticklocs()
    #print("y_loc: ", y_loc)
    #print("y_loc[1:-2]: ", y_loc[1:-2])
    #print("ylim: ", ax2.get_ylim())
    y2_min, y2_max = ax2.get_ylim()
    y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max - y2_min)*0.1]
    #print("y_loc: ", y_loc)
    ax2.set_yticks(y_loc)

    fig.tight_layout()
    fig.subplots_adjust(left=0.2)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_with_residuals"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    fig.clf()
    plt.close()

    ######## Make Pt_1000 Plot REALLY SHOULD BE ITS OWN FUNCTIONIN A CLASS
    R_Pt = avg_dict["R_Pt_1000"][~np.isnan(avg_dict["i"])]
    R_Pt_err = avg_dict["R_Pt_1000_err"][~np.isnan(avg_dict["i"])]

    r = np.corrcoef(residuals, R_Pt)
    print("Correlation matrix:", r)

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    print(R_Pt_err)
    print(R_Pt)
    ax1.errorbar(p_arr ,
             R_Pt,
             yerr = R_Pt_err,
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data" + " , correlation {:.3f}".format(r[0,1]))

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance Pt1000 [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname+ "Pt_1000"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    ##### Calculate Correlation: Should also be separate function

    return popt, pcov

def plot_poly_k(avg_dict, plotname):
    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]

    #HACK remove duplicates should they exist:
    i_arr = np.unique(i_arr)
    v_arr = np.unique(v_arr)
    v_err = np.unique(v_err)
    i_err = np.unique(i_err)


    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)
    #########
    # Fit original data with a 4th order polynomial, and then derive.
    # fit_func = lambda x,c0,c1,c2,c3,c4: (
    #     c4*x**4 + c3*x**3 + c2*x**2 + c1*x**1 + c0)

    fit_func = lambda x,c4,c3,c2,c1,c0: np.poly1d([c4, c3, c2, c1, c0])(x)
    popt, pcov = curve_fit(fit_func, p_arr, r_arr,  p0 = [0,0,0,0.135,66.7],
                           sigma = R_err(i_arr, i_err, v_arr, v_err),
                        #    absolute_sigma=True,
                        #    bounds = ((0.05,60,-1),(1.0,75,10))
    )
    #########Plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    plt.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label="fit"
                )

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #################### plot with residuals
    residuals = r_arr - fit_func(p_arr, *popt)
    chi2 = np.sum((residuals/r_err)**2)
    dof = len(residuals) - len(popt)
    chi2_red = chi2/dof

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    gs.update(#wspace=0.05
            hspace = 0.005
        )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ### ax1
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = r"data, $\chi^2_{red}$"+" = {:2.2f} ".format(chi2_red))

    # Plot Fit
    xdata=p_arr
    ax1.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label="fit"
                )

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    ax1.grid(True)
    ax1.legend(shadow=True, fontsize = 13)
    # ax1.tight_layout()

    #ax 2

    ax2.errorbar(p_arr ,
             residuals,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    ax2.plot(xdata, 0.0 * fit_func(p_arr, *popt), 'r-',
        #  label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [Ohm/µW ],
        #  R_0={:5.3f} $\pm$ {:2.1e} [Ohm]'''.format(
        #      popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
        #           ))
                )
    ax2.set_xlabel(r"power [µW]")
    ax2.set_ylabel(r"Residuals [$\Omega$]")

    ax2.grid(True)

    #make custom pruning of uppper tick (do not plot ticks in upper 10%)
    #so that ax2 tick does nto interfere with  ax1 tick
    ax2.locator_params(axis="y", min_n_ticks = 3
                       )
    y_loc = ax2.yaxis.get_majorticklocs()
    #print("y_loc: ", y_loc)
    #print("y_loc[1:-2]: ", y_loc[1:-2])
    #print("ylim: ", ax2.get_ylim())
    y2_min, y2_max = ax2.get_ylim()
    y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max - y2_min)*0.1]
    #print("y_loc: ", y_loc)
    ax2.set_yticks(y_loc)

    fig.tight_layout()
    fig.subplots_adjust(left=0.2)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_with_residuals"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    fig.clf()
    plt.close()

    ##########################
    def interval_m(i,j):
        m = (r_arr[j]-r_arr[i])/(p_arr[j]-p_arr[i])
        return m
    def interval_k(i,j):
        return 1/interval_m(i,j)

    def interval_k_err(i,j): 
        # TODO Check this
        delta_p = p_arr[j]-p_arr[i]
        delta_r = r_arr[j]-r_arr[i]
        s_p = np.sqrt(p_err[i]**2 + p_err[j]**2)
        s_r = np.sqrt(r_err[i]**2 + r_err[j]**2)
        k_err = np.sqrt(
            ((1 / delta_r) * s_p)**2
            + ((delta_p / delta_r ** 2) * s_r)**2
        )
        return k_err
    
    k_arr = np.array([interval_k(i,i+1) for i in range(len(p_arr)-1)])
    k_errs = np.array([interval_k_err(i,i+1) for i in range(len(p_arr)-1)])

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(r_arr[:-1] ,
             k_arr,
             yerr = k_errs,
             xerr = R_err(i_arr, i_err, v_arr, v_err)[:-1],
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Resistance [$\Omega$]")
    ax1.set_ylabel(r"k [µW/$\Omega$]")

    print("popt",popt)
    # Plot derivative based k
    xdata=r_arr
    plt.plot(xdata, 1/(np.poly1d([*popt]).deriv()(p_arr)), 'r-',
         label="k from deriv"
                )
    # Aditionally plot k derived from  fit to power over R rather than other


    
    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "deriv_k_over_R"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    return popt,pcov

def plot_poly_k_P_over_R(avg_dict, plotname):
    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]

    #HACK remove duplicates should they exist:
    i_arr = np.unique(i_arr)
    v_arr = np.unique(v_arr)
    v_err = np.unique(v_err)
    i_err = np.unique(i_err)


    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)
    #########
    # Fit original data with a 4th order polynomial, and then derive.
    # fit_func = lambda x,c0,c1,c2,c3,c4: (
    #     c4*x**4 + c3*x**3 + c2*x**2 + c1*x**1 + c0)

    fit_func = lambda x,c4,c3,c2,c1,c0: np.poly1d([c4, c3, c2, c1, c0])(x)
    popt, pcov = curve_fit(fit_func, r_arr, p_arr,  #p0 = [0,0,0,0.135,66.7],
                           sigma = P_err(i_arr, i_err, v_arr, v_err),
                        #    absolute_sigma=True,
                        #    bounds = ((0.05,60,-1),(1.0,75,10))
    )
    #########Plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(r_arr ,
             p_arr,
             yerr = P_err(i_arr, i_err, v_arr, v_err),
             xerr = R_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    plt.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label="fit"
                )

    #plt.xticks(rotation = 45)

    ax1.set_ylabel(r"power [µW]")
    ax1.set_xlabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #################### plot with residuals
    residuals = p_arr - fit_func(r_arr, *popt)
    chi2 = np.sum((residuals/p_err)**2)
    dof = len(residuals) - len(popt)
    chi2_red = chi2/dof

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    gs.update(#wspace=0.05
            hspace = 0.005
        )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ### ax1
    ax1.errorbar(r_arr ,
             p_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = r"data, $\chi^2_{red}$"+" = {:2.2f} ".format(chi2_red))

    # Plot Fit
    xdata=r_arr
    ax1.plot(xdata, fit_func(r_arr, *popt), 'r-',
         label="fit"
                )

    #plt.xticks(rotation = 45)

    ax1.set_ylabel(r"power [µW]")
    ax1.set_xlabel(r"Resistance [$\Omega$]")

    ax1.grid(True)
    ax1.legend(shadow=True, fontsize = 13)
    # ax1.tight_layout()

    #ax 2

    ax2.errorbar(r_arr ,
             residuals,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=r_arr
    ax2.plot(xdata, 0.0 * fit_func(r_arr, *popt), 'r-',
        #  label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [Ohm/µW ],
        #  R_0={:5.3f} $\pm$ {:2.1e} [Ohm]'''.format(
        #      popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
        #           ))
                )
    ax2.set_xlabel(r"Resistance [$\Omega$]")
    ax2.set_ylabel(r"Residuals [µW]")

    ax2.grid(True)

    #make custom pruning of uppper tick (do not plot ticks in upper 10%)
    #so that ax2 tick does nto interfere with  ax1 tick
    ax2.locator_params(axis="y", min_n_ticks = 3
                       )
    y_loc = ax2.yaxis.get_majorticklocs()
    #print("y_loc: ", y_loc)
    #print("y_loc[1:-2]: ", y_loc[1:-2])
    #print("ylim: ", ax2.get_ylim())
    y2_min, y2_max = ax2.get_ylim()
    y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max - y2_min)*0.1]
    #print("y_loc: ", y_loc)
    ax2.set_yticks(y_loc)

    fig.tight_layout()
    fig.subplots_adjust(left=0.2)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_with_residuals"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    fig.clf()
    plt.close()

    ##########################
    def interval_m(i,j):
        m = (r_arr[j]-r_arr[i])/(p_arr[j]-p_arr[i])
        return m
    def interval_k(i,j):
        return 1/interval_m(i,j)

    def interval_k_err(i,j): 
        # TODO Check this
        delta_p = p_arr[j]-p_arr[i]
        delta_r = r_arr[j]-r_arr[i]
        s_p = np.sqrt(p_err[i]**2 + p_err[j]**2)
        s_r = np.sqrt(r_err[i]**2 + r_err[j]**2)
        k_err = np.sqrt(
            ((1 / delta_r) * s_p)**2
            + ((delta_p / delta_r ** 2) * s_r)**2
        )
        return k_err
    
    k_arr = np.array([interval_k(i,i+1) for i in range(len(p_arr)-1)])
    k_errs = np.array([interval_k_err(i,i+1) for i in range(len(p_arr)-1)])

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(r_arr[:-1] ,
             k_arr,
             yerr = k_errs,
             xerr = R_err(i_arr, i_err, v_arr, v_err)[:-1],
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Resistance [$\Omega$]")
    ax1.set_ylabel(r"k [µW/$\Omega$]")


    # Plot derivative based k
    xdata=r_arr
    plt.plot(xdata, (np.poly1d([*popt]).deriv()(xdata)), 'r-',
         label="k from deriv"
                )
    # Aditionally plot k derived from  fit to power over R rather than other


    
    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "deriv_k_over_R"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # Aditionally plot k derived from  fit to power over R rather than other
    # way around


    return popt,pcov


def plot_interval_k(avg_dict, plotname):

    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]

    #HACK remove duplicates should they exist:
    i_arr = np.unique(i_arr)
    v_arr = np.unique(v_arr)
    v_err = np.unique(v_err)
    i_err = np.unique(i_err)


    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)

    def interval_m(i,j):
        m = (r_arr[j]-r_arr[i])/(p_arr[j]-p_arr[i])
        return m
    def interval_k(i,j):
        return 1/interval_m(i,j)

    def interval_k_err(i,j): 
        # TODO Check this
        delta_p = p_arr[j]-p_arr[i]
        delta_r = r_arr[j]-r_arr[i]
        s_p = np.sqrt(p_err[i]**2 + p_err[j]**2)
        s_r = np.sqrt(r_err[i]**2 + r_err[j]**2)
        k_err = np.sqrt(
            ((1 / delta_r) * s_p)**2
            + ((delta_p / delta_r ** 2) * s_r)**2
        )
        return k_err

    

    # #########
    # def fit_func(P, m, R_0):
    #     R = R_0 + m * P
    #     return R
    m_arr = np.array([interval_m(i,i+1) for i in range(len(p_arr)-1)])
    k_arr = np.array([interval_k(i,i+1) for i in range(len(p_arr)-1)])
    k_errs = np.array([interval_k_err(i,i+1) for i in range(len(p_arr)-1)])

    #########Plot k
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(p_arr[:-1] ,
             k_arr,
             yerr = k_errs,
             xerr = P_err(i_arr, i_err, v_arr, v_err)[:-1],
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"k [µW/$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #########Plot k over R
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(r_arr[:-1] ,
             k_arr,
             yerr = k_errs,
             xerr = R_err(i_arr, i_err, v_arr, v_err)[:-1],
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Resistance [$\Omega$]")
    ax1.set_ylabel(r"k [µW/$\Omega$]")
    # Plot mmovinng average:
    # def moving_average(a, n=7):
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n

    def weighted_moving_average(a,errs, n=4):
        # double sided n: the point itself and n forward and back
        # Weight with 1 / variance
        ret = [np.average(a[i-n : i+n+1], weights = 1/(errs[i-n : i+n+1])**2) 
               for i in range(n,len(a)-n)]
        return np.array(ret)
    #overwrite function
    moving_average = weighted_moving_average
    k_avg_arr = moving_average(k_arr, errs = k_errs)
    r_avg_arr = moving_average(r_arr[:-1] , errs= r_err[:-1])
    ax1.plot(r_avg_arr, k_avg_arr, 'r-',
         label="weighted moving average"
                )

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_over_R"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #########Plot k over R with B-spline
    # NOTE also consider "UnivariateSpline" for the same job.
    def B_spline_fit(x, y, degree = 3):
        k = degree
        from scipy.interpolate import make_lsq_spline, BSpline
        # 7 - 2 = 5 uniform knots inside data range with 2 endpoint knnots
        t  = np.linspace(x[0], x[-1], num=7, endpoint=True)
        #t = np.arange[-1, 0, 1]
        # concatenate with internal knots
        # add k boundary knots 
        # to get k + 1 boundary knots on both sides to make "regular"
        # following: 
        # https://scipy.github.io/devdocs/reference/generated/
        # scipy.interpolate.make_lsq_spline.html
        t = np.r_[(x[0],)*(k),
                t,
                (x[-1],)*(k)]
        spl = make_lsq_spline(x, y, t, k) 
        return spl

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(r_arr[:-1] ,
             k_arr,
             yerr = k_errs,
             xerr = R_err(i_arr, i_err, v_arr, v_err)[:-1],
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"Resistance [$\Omega$]")
    ax1.set_ylabel(r"k [µW/$\Omega$]")

    moving_average = weighted_moving_average
    k_avg_arr = moving_average(k_arr, errs = k_errs)
    r_avg_arr = moving_average(r_arr[:-1] , errs= r_err[:-1])
    ax1.plot(r_avg_arr, k_avg_arr, 'r-',
         label="weighted moving average"
                )
    
    xs = np.linspace(r_arr[:-1][0], r_arr[:-1][-1], num=50, endpoint=True)
    # print("r_arr[:-1]",r_arr[:-1])

    #r_arr is not sorted (likely duee to noise) but it has to be 
    r_sorted = np.array(sorted(r_arr[:-1]))
  
    k_sorted = k_arr[[np.argwhere(r_arr == r)[0][0] for r in r_sorted]]
    #print("r_sorted", r_sorted)


    xs = np.linspace(r_sorted[0], r_sorted[-1], num=50, endpoint=True)
    spl = B_spline_fit(r_sorted, k_sorted)
    plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ B-spline')
    #TODO add weights to spline fit

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_over_R_B-spline"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    
    #########Plot m
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(p_arr[:-1] ,
             m_arr,
            #  yerr = R_err(i_arr, i_err, v_arr, v_err),
            #  xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"m [$\Omega$/µW]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_m"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    # return B_spline
    return spl

    

def P14_R_over_P_calib(avg_dict, plotname):
    # fit 0.9mA to 1.1mA range directly

    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]
    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)
    # #fit
    #   fit_func
    def fit_func(P, m, R_0, a):
        R = R_0 + m * P + a* P**(1/4)
        return R

    popt, pcov = curve_fit(fit_func, p_arr, r_arr, p0 = [0.135,66.7, 0.0],
                           sigma = R_err(i_arr, i_err, v_arr, v_err),
                           absolute_sigma=True,
                           bounds = ((0.05,60,-1),(1.0,75,10))
    )


    #########Plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    label=(("fit: m={:5.4f} $\pm$ {:2.1e} [$\Omega$/µW ],".format(
             popt[0], np.sqrt(pcov[0][0]))
          +" \n R_0={:5.3f} $\pm$ {:2.1e} [$\Omega$/µW]".format(
              popt[1], np.sqrt(pcov[1][1])) 
          +" \n a={:5.3f} $\pm$ {:2.1e} [$\Omega$/(µW)^(1/4)]".format(
              popt[2], np.sqrt(pcov[2][2])) 
                  ))
    plt.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label=label
                )

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #################### plot with residuals
    residuals = r_arr - fit_func(p_arr, *popt)
    chi2 = np.sum((residuals/r_err)**2)
    dof = len(residuals) - len(popt)
    chi2_red = chi2/dof

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    gs.update(#wspace=0.05
            hspace = 0.005
        )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ### ax1
    ax1.errorbar(p_arr ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = r"data, $\chi^2_{red}$"+" = {:2.2f} ".format(chi2_red))

    # Plot Fit
    xdata=p_arr
    label=(("fit: m={:5.4f} $\pm$ {:2.1e} [$\Omega$/µW ],".format(
             popt[0], np.sqrt(pcov[0][0]))
          +" \n R_0={:5.3f} $\pm$ {:2.1e} [$\Omega$/µW]".format(
              popt[1], np.sqrt(pcov[1][1])) 
          +" \n a={:5.3f} $\pm$ {:2.1e} [$\Omega$/(µW)^(1/4)]".format(
              popt[2], np.sqrt(pcov[2][2])) 
                  ))
    ax1.plot(xdata, fit_func(p_arr, *popt), 'r-',
         label=label)

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    ax1.grid(True)
    ax1.legend(shadow=True, fontsize = 13)
    # ax1.tight_layout()

    #ax 2

    ax2.errorbar(p_arr ,
             residuals,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")

    # Plot Fit
    xdata=p_arr
    ax2.plot(xdata, 0.0 * fit_func(p_arr, *popt), 'r-',
        #  label=(('''fit: m={:5.4f} $\pm$ {:2.1e} [Ohm/µW ],
        #  R_0={:5.3f} $\pm$ {:2.1e} [Ohm]'''.format(
        #      popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])) 
        #           ))
                )
    ax2.set_xlabel(r"power [µW]")
    ax2.set_ylabel(r"Residuals [$\Omega$]")

    ax2.grid(True)

    #make custom pruning of uppper tick (do not plot ticks in upper 10%)
    #so that ax2 tick does nto interfere with  ax1 tick
    ax2.locator_params(axis="y", min_n_ticks = 3
                       )
    y_loc = ax2.yaxis.get_majorticklocs()
    #print("y_loc: ", y_loc)
    #print("y_loc[1:-2]: ", y_loc[1:-2])
    #print("ylim: ", ax2.get_ylim())
    y2_min, y2_max = ax2.get_ylim()
    y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max - y2_min)*0.1]
    #print("y_loc: ", y_loc)
    ax2.set_yticks(y_loc)

    fig.tight_layout()
    fig.subplots_adjust(left=0.2)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_with_residuals"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    fig.clf()
    plt.close()

    ######## Make Pt_1000 Plot REALLY SHOULD BE ITS OWN FUNCTIONIN A CLASS
    R_Pt = avg_dict["R_Pt_1000"][~np.isnan(avg_dict["i"])]
    R_Pt_err = avg_dict["R_Pt_1000_err"][~np.isnan(avg_dict["i"])]

    r = np.corrcoef(residuals, R_Pt)
    print("Correlation matrix:", r)

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    print(R_Pt_err)
    print(R_Pt)
    ax1.errorbar(p_arr ,
             R_Pt,
             yerr = R_Pt_err,
             xerr = P_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data" + " , correlation {:.3f}".format(r[0,1]))

    #plt.xticks(rotation = 45)

    ax1.set_xlabel(r"power [µW]")
    ax1.set_ylabel(r"Resistance Pt1000 [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname+ "Pt_1000"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    ##### Calculate Correlation: Should also be separate function

    return popt, pcov

def plot_avg_over_index(avg_dict, plotname):

    i_arr = avg_dict["i"][~np.isnan(avg_dict["i"])]
    #print("np.isnan(avg_dict['i'])",np.isnan(avg_dict["i"]))
    v_arr = avg_dict["v"][~np.isnan(avg_dict["i"])]
    v_err = avg_dict["v_err"][~np.isnan(avg_dict["i"])]
    i_err = avg_dict["i_err"][~np.isnan(avg_dict["i"])]
    p_arr = i_arr * v_arr
    r_arr = v_arr / i_arr
    r_err = R_err(i_arr, i_err, v_arr, v_err)
    p_err = P_err(i_arr, i_err, v_arr, v_err)
    #fit
    #########Plot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar([i for i in range(len(r_arr))] ,
             r_arr,
             yerr = R_err(i_arr, i_err, v_arr, v_err),
             fmt = ".",
             # markersize=1,
             label = f"data")


    ax1.set_xlabel(r"Index")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    ######### Plot just 0.9 to 1mA corresponding
    r_arr_cut = r_arr[18:36]

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    # # with errorbars
    ax1.errorbar([i for i in range(len(r_arr_cut))] ,
             r_arr_cut,
             yerr = R_err(i_arr, i_err, v_arr, v_err)[18:36],
             fmt = ".",
             # markersize=1,
             label = f"data")


    ax1.set_xlabel(r"Index")
    ax1.set_ylabel(r"Resistance [$\Omega$]")

    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname + "_cut"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    return

if __name__ =="__main__": 
################## 2022-12-08 New Automated Calib run 2022-12-09_calib_7
    run_name = "2023-01-09_calib_long"
    prep_data_calib(filename  = "2023-01-09_calib_long.json"
                     , dataname = run_name, data_dir = data_dir
                     )
    data_dict = load_data(run_name, data_dir = data_dir)

    #print('data_dict["R_Pt_1000"][10:-10]',data_dict["R_Pt_1000"][10:-10])
    out_dir = os.sep + run_name + os.sep
    os.makedirs(plot_dir + out_dir, exist_ok=True)

    ##### See if old scripts will just work
    avg_dict = raw_dict_to_avg(data_dict)
    avg_dict_low = raw_dict_to_avg(data_dict,
            key_list = ["0.00" + "{:0>3d}".format(1*i) for i in range(1,10)] 
                        #+  [str(0.1*i) for i in range(2,3)]
                        )

    plot_R_vs_P(avg_dict, plotname = out_dir + "R_vs_P")
    plot_R_vs_P(avg_dict_low, plotname = out_dir +"R_vs_P_to_0.1")

    fit_base_R(avg_dict_low, out_dir +"R_vs_P_offset_fit")
    #calib around 1mA
    avg_dict_09to11 = raw_dict_to_avg(data_dict,
                    key_list = ["0.00" + "{:0>3d}".format(90 + 1*i) 
                                for i in range(1,10)] 
                             + ["0.00" + "{:0>3d}".format(100 + 1*i) 
                                for i in range(1,10)] 
                             )
    avg_dict_09to116 = raw_dict_to_avg(data_dict,
                    key_list = ["0.00" + "{:0>3d}".format(90 + 1*i) 
                                for i in range(1,10)] 
                             + ["0.00" + "{:0>3d}".format(100 + 1*i) 
                                for i in range(1,10)] 
                             + ["0.00" + "{:0>3d}".format(100 + 1*i) 
                                for i in range(1,61)] 
                             )

    avg_dict_14to16 = raw_dict_to_avg(data_dict,
                    key_list =["0.00" + "{:0>3d}".format(100 + 1*i) 
                                for i in range(40,61)] 
                             )
    avg_dict_ref = raw_dict_to_avg(data_dict,
                    key_list = ["0.00" + "{:0>3d}".format(100) 
                                for i in range(1)] 
                             )

    # plot_avg_over_index(avg_dict_ref, out_dir +"R_over index_fit_i_ref")
    # basic_R_over_P_calib(avg_dict_09to11, out_dir +"R_vs_P_fit_09to11")
    # basic_R_over_P_calib(avg_dict_09to116, out_dir +"R_vs_P_fit_09to116")
    # basic_R_over_P_calib(avg_dict_14to16, out_dir +"R_vs_P_fit_14to16")
    # out_dir_2 = out_dir + os.sep + "4_root_P" + os.sep
    # os.makedirs(plot_dir + out_dir_2, exist_ok=True)
    # P14_R_over_P_calib(avg_dict_09to11, out_dir_2 +"R_vs_P_fit_09to11")
    # P14_R_over_P_calib(avg_dict_09to116, out_dir_2 +"R_vs_P_fit_09to116")
    # P14_R_over_P_calib(avg_dict_14to16, out_dir_2 +"R_vs_P_fit_14to16")

    # plot_calib_sectioned(data_dict, out_dir + "plot_calib")

    #####
    #Plot interval slopes
    #plot_interval_k(avg_dict_09to116, out_dir +"interval_k_09to116")
    plot_poly_k(avg_dict_09to116, out_dir +"poly_fit_09to116")
    plot_poly_k_P_over_R(avg_dict_09to116, out_dir +"PoR_poly_fit_09to116")

