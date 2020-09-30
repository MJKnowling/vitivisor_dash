# this is a dashboard-tailored version of the main worker script in the repo https://gitlab.com/mjknowling/da_opt_vinelogic
# TODO: will only include relevant functions when pushing to public dash repo

import os
import sys
import shutil
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import time

font = {'size': 30}
import matplotlib

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad': '10'}
pylab.rcParams.update(params)
figsize = (8,6)

matplotlib.rc("font", **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
colors = ['b', 'r', 'c']  #plt.rcParams['axes.prop_cycle'].by_key()['color']

import pyemu

# globals
base_model_ws = os.path.join("_base_model")  # this should not be changed  #("low_irrig")
new_model_ws = os.path.join("_temp_lrc_block47")
if not os.path.exists(new_model_ws):
    os.mkdir(new_model_ws)

if not os.path.exists(os.path.join("plots")):
    os.mkdir("plots")

num_workers = 4

frun_vinelogic = os.path.join("run_vinelogic.R")
vines_out_fname = "vines_output.csv"
vines_summ_fname = "vines_summary.csv"
vinelogic_nan = -99
weather_input_fname = "MILD.csv"
irrig_input_fname = "ObservedIrrigation.csv"
forecasts = ["1_yield", "1_ffindex", "1_doy06", "1_doy08", "1_doy01", "1_doy02"]
states = ["LAI", "total_soil_water", "FFulindex", "BerryDw", "daily_thermal_time",
          "cumul_thermal_time", "Berry_thermal_time", "Matdtt", "Cpool", "shoot_num",
          "berry_num_shoot", "berry_num_vine", "irrigation"]

start_date = datetime.strptime("2019-01-01", '%Y-%m-%d')  # from frun_vinelogic
daily_time_steps = 900  # from frun_vinelogic

pst_fname = "pest.pst"
#obs_types = ["lai", "soil_water_stress"]  # TODO: combine obs into truth_obs_data = {type: [res, noise]}
obs_dates = None
obs_times = 14  #every th
#obs_weight = 5.0
#obs_dates = ["1999-01-01", "2000-09-15", "2000-12-01"]  #["2000-09-01", "2000-10-01", "2000-11-01", "2000-12-01", "2001-01-01"]
obs_dict = {"lai": 5.0}#, "soil_water_stress": 10.0}


def ts_plot_helper(cwd, fname="state_time_series.pdf"):
    # plot state variable time series from out csv
    df = pd.read_csv(os.path.join(cwd, vines_out_fname),
                     index_col="DayOfYear")
    vars_of_interest = ["FFulindex", "LAI", "BerryDw", "soil_water_stress", "irrigation"]
    fig, ax = plt.subplots(len(vars_of_interest), sharex=True)
    for i, ax in enumerate(ax):
        df.loc[:, vars_of_interest[i]].plot(ax=ax)
        ax.set_ylabel(vars_of_interest[i])
    plt.savefig(os.path.join("plots", fname))
    plt.close()

def mm_to_ML_irrig(mm):
    # variables  # TODO: declare before
    # LRC block 47 area (email from Ryan Tan and accounting for drip line only around line
    block_rowwise_dist = 89  # m
    vine_rows = 18
    irrig_width = 2.5  # m  # TODO: align with zone approach in VineLOGIC
    irrig_area = block_rowwise_dist * (irrig_width * vine_rows)
    #TODO: temp ************************************************
    irrig_area = irrig_area * 2.2031  # scale to 1 ha
    #TODO: temp ************************************************
    #irrig_area = block_rowwise_dist * 51  # total area in m2
    total_ML = mm * 0.001 * irrig_area * 0.001
    #print(mm, total_ML, mm * 0.001 * (block_rowwise_dist * 51) * 0.001)
        #TODO: temp ************************************************
    return total_ML, total_ML / (block_rowwise_dist * 51 * 0.0001 * 2.2031)  # tota, total per ha
        #TODO: temp ************************************************

def ts_compare_irrig_plot(cwds, which, d, show_plot=True, total=False):
    if not isinstance(cwds, list):  # dict
        dd = cwds.copy()
        cwds = list(cwds.keys())
    if which == "lai":  # yuck!
        which = "LAI"
    elif which == "fruit":
        which = "FruitDw"
    elif which == "supplydemand":
        which = ["FruitSink", "Cpool"]
    if which == "soil_water":
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)
        axs = np.array(axs)
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    q_irr_scen, lai_irr_scen = {}, {}
    q_irr_per_ha_scen, q_mm_total = {}, {}
    rain_season_total, rain_annual_total = {}, {}
    #for i, ax in enumerate(ax):
    dfs, dfs_ref, dfs_all = {}, {}, {}
    yl = []
    #if "base" in cwds[1]:
     #   cwds.reverse()
    #print(d)
    for i, scen_ws in enumerate(cwds):
        df = pd.read_csv(os.path.join(scen_ws, vines_out_fname),
                         index_col="DayOfYear")
        dates = pd.date_range(start_date, periods=daily_time_steps)
        df.index = dates
        if which == "rain":
            df_all = df.copy()
            df_all = df.loc[datetime.strftime(start_date + timedelta(365 + 365/2 - 1), '%Y-%m-%d'):
                            datetime.strftime(start_date + timedelta(365 + 365*1.5 - 1), '%Y-%m-%d'), :]
            df_all = df_all.loc[:, which]
        df = df.loc[datetime.strftime(start_date + timedelta(365 + (365 / 2) + 61), '%Y-%m-%d'):
                    datetime.strftime(start_date + timedelta(365*2 + (365/2)), '%Y-%m-%d'), :] #timedelta(900), '%Y-%m-%d'), :]
        
        # TODO:         
        '''summ_df = pd.read_csv(os.path.join(scen_ws, vines_summ_fname))
        bb = summ_df.loc[:, "DOY02"][0]
        hv = summ_df.loc[:, "DOY06"][0]
        bb = doy_to_date(bb, (start_date + timedelta(365)).year).strftime("%d %b %Y")
        hv = doy_to_date(hv, (start_date + timedelta(365*3)).year).strftime("%d %b %Y")
        
        #df = df.loc[datetime.strftime(start_date + timedelta(365 + (365 / 2) + 61), '%Y-%m-%d'):
         #           datetime.strftime(start_date + timedelta(365*2 + (365/2)), '%Y-%m-%d'), :] #timedelta(900), '%Y-%m-%d'), :]
        df = df.loc[bb:hv, :]
        '''

        if which == "ATheta":# or which == "soil_water":
            df_ref = df.loc[:, "IRCRITSW"]
            dfs_ref[scen_ws] = df_ref
        if which == "soil_water":
            sw_states = ["soil_water_top", "soil_water_mid", "soil_water_bot"]
            df = df.loc[:, sw_states]
        else:
            yl.append(ax.get_ylim()[1])
            xl = ax.get_xlim()[1]
            df = df.loc[:, which]
        dfs[scen_ws] = df
        if which == "rain" and total is True:
            dfs_all[scen_ws] = df_all
        if "irrigation" in which:
            q, q_per_ha = mm_to_ML_irrig(df.sum())  # compute sum in ML
            #print(df.loc[df.values>0.0])
            #print(df.sum(), q_per_ha)
            #ax.text(xl * 0.5, yl * 0.8, "Total_{0} = {1:.2f} ML/ha ({2} mm)".format(scen_ws, q_per_ha, int(q_per_ha * 100)))
            q_irr_scen[scen_ws] = q
            q_mm_total[scen_ws] = df.sum()
            q_irr_per_ha_scen[scen_ws] = q_per_ha
            ax.set_ylabel(which.title() + "(mm/day)")
        elif which == "rain":
            if total is True:
                rain_season_total[scen_ws] = df.sum()  # seasonal
                rain_annual_total[scen_ws] = df_all.sum()  # annual - take hydrological year
            ax.set_ylabel("Rainfall (mm/day)")
        elif which != "soil_water":
            if which.lower() == "LAI".lower():
                lai_irr_scen[scen_ws] = df
                ax.set_ylabel("Canopy Density\n(Leaf Area Index)")
            elif which.lower() == "FruitDw".lower():
                ax.set_ylabel("Fruit (Dry) Weight\n(per vine) (grams)")
            elif which.lower() == "Brix".lower():
                ax.set_ylabel("Brix\n(degrees)")
            elif which == "infiltration":
                ax.set_ylabel("{} (cm/day)".format(which.title()))
            elif which == "evap":
                ax.set_ylabel("Evaporation\n(surface and upper soil)\n(cm/day)")
            elif which == "root_uptake":
                ax.set_ylabel("{} (cm/day)".format(which.replace("_", " ").title()))
            #elif which == "soil_water":
             #   ax.set_ylabel("Soil saturation (-)")
            elif which == "drainage":
                ax.set_ylabel("Drainage (cm/day)")
            elif which == "ATheta":
                ax.set_ylabel("Irrigation Trigger-Relevant\nSoil Water\n")#(= $\Sigma^{nlayr_irri_depthj}_{1}$ (Sw - Wilting Point) * dz / Sigma^{nlayr}_{1}$ (Field Capacity - Wilting Point) * dz")
            elif which == "soil_water_balance":
                ax.set_ylabel("Water Balance Error (cm)")#\n(sum(total soil water, Rain, Irrig, Pond)) - (sum(TSW2, pond, runoff, drain, ES, TRU))")

            elif which == "ponding":
                ax.set_ylabel("Ponding (cm)")
            elif which == "runoff":
                ax.set_ylabel("Surface Runoff (cm/day)")

            elif which == "soil_water_stress1":
                ax.set_ylabel("{} Index (-)".format(which.replace("_", " ").title().strip("1")))
            elif which == "Cpool":
                ax.set_ylabel("Available Energy For \nFruit Development (FIND UNITS)")
            elif which == "FruitSink":
                ax.set_ylabel("Fruit Development \nEnergy Sink (FIND UNITS)")
            elif which == "supplydemand":
                ax.set_ylabel("Fruit Energy Demand versus Available Energy")
            elif which == "VineEop":
                ax.set_ylabel("Potential Vine ET (cm/day)")
            elif which == "Tru":
                ax.set_ylabel("Root Uptake (cm/day)")
            elif which == "total_soil_water1":
                ax.set_ylabel("Total Soil Water (cm)")
    if which != "soil_water":
        dfs = pd.DataFrame(dfs)
    if which == "ATheta":# or which == "soil_water":
        dfs_ref = pd.DataFrame(dfs_ref)
    if which == "rain" and total is True:
        dfs_all = pd.DataFrame(dfs_all)
    #if which == "irrigation":
        #print(dfs.head())
        #ind = np.arange(len(dfs))
        #width = 0.1
        #r1 = ax.bar(ind + width / 2, dfs["irrig_base"], width, label="Base", color='#1f77b4')
        #r2 = ax.bar(ind + width / 2, dfs["irrig_low"], width, label="Low", color='#ff7f0e')
        #ax.set_ylabel('Scores')
        #ax.set_title('Scores by group and gender')
        #ax.set_xticks(ind)
        #ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
        #ax.legend()

        #dfs.plot(kind="bar", ax=ax, alpha=1.0)
        #skip = 31
        #ticklabels = [''] * len(dfs)
        #ticklabels[::skip] = dfs.index[::skip].strftime('%b %Y')
        #ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
        #ax.xaxis.set_minor_formatter(mticker.FixedFormatter(ticklabels))
    #else:
    if ("irrigation" in which and total is True) or ("rain" in which and total is True):
        text_height_irr, text_color, text_height_rain = [0.9, 0.7, 0.5], [colors[x] for x in range(3)], [0.9, 0.6, 0.3]
        #print(q_irr_per_ha_scen)
        for i, scen_ws in enumerate(cwds):
            if "irrigation" in which:
                ax.text(x=xl / 2, y=max(yl) * text_height_irr[i], 
                    s="Seasonal Irrigation Total ({0}):\n {1:.1f} mm ({2:.1f} ML/ha)".format(dd[scen_ws], q_mm_total[scen_ws], q_irr_per_ha_scen[scen_ws]), 
                    ha='center', va='center', fontsize=24, color=text_color[i])
            elif "rain" in which:
                ax.text(x=xl / 2, y=max(yl) * text_height_rain[i], 
                    s="Rainfall Totals ({0}):\n  Seasonal (Sep-May): {1:.1f} mm\n  Annual (Jul-Jun): {2:.1f} mm".format(dd[scen_ws], rain_season_total[scen_ws], rain_annual_total[scen_ws]), 
                    ha='center', va='center', fontsize=24, color=text_color[i])
            ax.axis('off')
    else:
        #dfs.plot(ax=ax, alpha=1.0, lw=2)
        for i, scen_ws in enumerate(cwds):
            if which == "supplydemand":
                # TODO
                pass
            elif which == "soil_water":
                sp = pd.read_json(os.path.join("_base", "SoilProfile.json"))
                vs = {"SLLL": "wilting point", "SDUL": "field capacity", "SSAT": "saturation"}
                zs = {0: [0, 5], 1: [7, 50], 2: [13, 200]}  # lay and depth pairs
                for ii, ax in enumerate(axs.reshape(-1)):
                    dfs[scen_ws][sw_states[ii]].plot(ax=ax, alpha=1.0, lw=2, color=colors[i])
                    xlim = ax.get_xlim()
                    ax.set_ylabel("Soil Water\n Content (-)\n At Depth \n{0} cm".format(str(zs[ii][1])))
                    ax.set_ylim(0, 0.5)
                    if ii == ((axs.shape[0]) - 1):
                        ax.set_xlabel("Date")
                    if i == len(cwds) - 1:  # these are fixed variables
                        for vi in vs.items():
                            #v = np.mean(sp["SoilLayerProperties"][vi[0]]["Value"])
                            v = sp["SoilLayerProperties"][vi[0]]["Value"][zs[ii][0]]
                            ax.axhline(y=v, linewidth=2, linestyle='--', color='k', alpha=0.8)
                            props = dict(boxstyle='square', facecolor='white', alpha=0.3, edgecolor='none')
                            if vi[0] == "SDUL":
                                _x = xlim[0] + (0.05 * (xlim[1] - xlim[0]))
                            else:
                                _x = xlim[1] - (0.2 * (xlim[1] - xlim[0]))
                            ax.text(x=_x, y=v + 0.005, 
                                    s="{}".format(vi[1].title()), fontsize=12, alpha=1.0, bbox=props)
                    #ax.set_xlim(xlim)
            else:
                dfs[scen_ws].plot(ax=ax, alpha=1.0, lw=2, color=colors[i])
                xlim = ax.get_xlim()
                #print(xlim)
        if which == "soil_water":
            l = [x for x in dfs.keys()]
        else:
            l = [x for x in dfs.columns]
        for i, scen_ws in enumerate(cwds):
            if which == "ATheta":# or which == "soil_water": 
                dfs_ref[scen_ws].plot(ax=ax, alpha=1.0, lw=2, color=colors[i], linestyle='--')
        if "irrig" in which:
            ax.legend([dd[x] for x in l], loc='upper right')
        elif which != "soil_water":
            ax.legend([dd[x] for x in l])
            ax.set_xlabel("Date")
    #plt.savefig(os.path.join("plots", "ts_{}.pdf".format(which)))
    if not show_plot is True:
        plt.close()

    return q_irr_scen, lai_irr_scen, (fig, ax)

def kg_per_ha_to_tonnes(kg_per_ha):
    # variables  # TODO: declare before
    # LRC block 47 area (email from Ryan Tan and accounting for drip line only around line
    block_area = 89 * 51 * 0.0001  # ha
    #TODO: temp ************************************************
    block_area = block_area * (1 / block_area)  # scale to 1 ha
    #print(block_area)
    #TODO: temp ************************************************
    return kg_per_ha * block_area * 0.001

def yield_revenue_compare(cwds, which, d, show_plot=True):
    if not isinstance(cwds, list):  # dict
        dd = cwds.copy()
        cwds = list(cwds.keys())
    # variables  # TODO: declare
    #print(d.keys())
    #grape_price = 697.0  # $; 2019/2020 FY Riverland Shiraz grape price (see pg SA4 WA SA2020 report)

    block_rowwise_dist = 89  # m
    block_acrossrow_dist = 51  # m
    block_area = block_rowwise_dist * block_acrossrow_dist  # m**2
    #TODO: temp ************************************************
    block_area = block_area * (1 / block_area)  # scale to 1 ha
    #TODO: temp ************************************************
    block_area_ha = block_area * 0.0001

    # yield bar plot
    yields = {}
    for scen_ws in cwds:
        yields[scen_ws] = kg_per_ha_to_tonnes(pd.read_csv(os.path.join(scen_ws, vines_summ_fname))
                                              .loc[:, "Yield"][0])

    # revenue (using grape price)
    yield_rev = yields.copy()
    #print(yield_rev, scen_ws, d[scen_ws]['grape_price'])
    #print("yield rev {}".format(yield_rev))
    #print("d {}".format(d))
    #print("dd {}".format(dd))
    yield_rev = {x: a * d[dd[x]]['grape_price'] for (x, a) in yield_rev.items()}
    #print(yield_rev)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    if which == "yield":
        keys = yields.keys()
        values = yields.values()
        # TODO: bar only if num_scen > 1!
        bl = ax.bar(keys, values)
        for i, k in enumerate(keys):
            #print(k)
            #if i == 0:
            #if "base" not in k.lower():
             #   bl[i].set_color('#ff7f0e')
            bl[i].set_color(colors[i])
            #ax.text(i, ax.get_ylim()[1] / 2, 
             #   "Tonnes per ha:\n {0:.2f}".format(yields[k] / block_area_ha), size=14, ha='center', alpha=0.8)
        #print([tick for tick in plt.gca().get_xticklabels()])
        ax.set_xticklabels([dd[x].split("_")[-1] for x in keys])
        plt.ylabel("Harvest Yield (Tonnes/ha)")  #**tmp**
        #plt.savefig(os.path.join("plots", "yield_irrig_scen.pdf"))
        if not show_plot is True:
            plt.close()
    elif which == "revenue":
        keys = yield_rev.keys()
        values = yield_rev.values()
        bl = ax.bar(keys, values)
        for i, k in enumerate(keys):
            #if i != 0:  #if "base" not in k.lower():
             #   bl[i].set_color('#ff7f0e')
            bl[i].set_color(colors[i])
        ax.set_xticklabels([dd[x].split("_")[-1] for x in keys])
        plt.ylabel("Harvest Revenue ($/ha)")  #**tmp**
        #plt.savefig(os.path.join("plots", "yield_revenue_irrig_scen.pdf"))
        if not show_plot is True:
            plt.close()

    return yield_rev, (fig, ax)


def irrig_compare(q_irr_scen, d, mapper, show_plot=True):

    # variables  # TODO: pass as args to func
    #water_market_rate = 401  # $/ML - https://www.waterconnect.sa.gov.au/Systems/WTR/Pages/water-trades-allocation-charts.aspx
    #water_delivery_rate = 60  # $/ML - http://www.cit.org.au:84/Downloads/CIT%20Water%20Prices%2019%2020%20as%20at%201%20October%202019.pdf
    #variable_cost_per_l = 0.15
    #fixed_cost_per_l = 0.08
    #sale_price_per_l = 0.03
    #allocation = min(q_irr_scen.values()) * factor_area_units  #max(q_irr_scen.values()) * factor_area_units * 1.2

    # but first total irr vol bar chart
    #keys = q_irr_scen.keys()
    #values = q_irr_scen.values()
    #plt.bar(keys, values)
    #plt.ylabel("sum irrigation (ML)")
    #plt.savefig("sum_irrig.pdf")
    #plt.close()

    # cost
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    dolla_irr_scen = q_irr_scen.copy()

    #print(dolla_irr_scen)
    #print(d)

    dolla_irr_scen = {x: (a * d[mapper[x]]['water_delivery_rate'] if dolla_irr_scen[x] <= d[mapper[x]]['water_entitlement']
                          else a * d[mapper[x]]['water_delivery_rate'] + ((dolla_irr_scen[x] - d[mapper[x]]['water_entitlement']) * d[mapper[x]]['water_market_rate']))
                          for x, a in dolla_irr_scen.items()
                          }
    #print(dolla_irr_scen)
    #tmp = 89 * 51 * 0.0001 * 2.2031
    #dolla_irr_scen = {x: a * tmp for x, a in dolla_irr_scen.items()} #**tmp**

    keys = dolla_irr_scen.keys()
    values = dolla_irr_scen.values()
    b = ax.bar(keys, values)
    for i, k in enumerate(keys):
        #if i != 0:#if "base" not in k.lower():
         #   b[i].set_color('#ff7f0e')
        b[i].set_color(colors[i])
    #print(d, mapper, keys)
    ax.set_xticklabels([mapper[x].split("_")[-1] for x in keys])
    plt.ylabel("Irrigation Cost ($/ha)")  #**tmp**
    #plt.savefig(os.path.join("plots", "cost_irrig_scen.pdf"))
    if not show_plot is True:
        plt.close()

    # sell
    #dolla_sell_irr_scen = q_irr_scen.copy()
    #factor = allocation - (factor_area_units * cost_consump_per_l)
    #dolla_irr_scen = {x: a * factor for (x, a) in dolla_sell_irr_scen.items()}

    return dolla_irr_scen, (fig, ax)


def gross_margin(irrig_cost, grape_revenue, which, d, mapper, spray_cost=0.0, tip_cost=0.0, include_canopy_mgmt=True,
                 include_disease_mgmt=True):
    revenue = grape_revenue
    cost = irrig_cost
    if isinstance(spray_cost, dict) and include_disease_mgmt:
        cost = {key: cost[key] + spray_cost.get(key, 0) for key in cost}
    if isinstance(tip_cost, dict) and include_canopy_mgmt:
        cost = {key: cost[key] + tip_cost.get(key, 0) for key in cost}
    
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    if which == "gross_margin":
        gross_margin = {key: revenue[key] - cost.get(key, 0) for key in revenue}
        keys = gross_margin.keys()
        values = gross_margin.values()
        b = ax.bar(keys, values)
        for i, k in enumerate(keys):
            #if i != 0:#if "base" not in k.lower():
             #   b[i].set_color('#ff7f0e')
            b[i].set_color(colors[i])
        ax.set_xticklabels([mapper[x].split("_")[-1] for x in keys])
        plt.ylabel("Gross Margin ($)")
        #plt.savefig(os.path.join("plots", "gross_margin.pdf"))
        #plt.close()
    elif which == "cost_contribs":
        # plot cost contribs
        ds = [irrig_cost, spray_cost, tip_cost]
        all_costs = {}
        for k in cost.keys():
            all_costs[k] = [x[k] for x in ds]
        c, v = [], []
        for key, val in all_costs.items():
            c.append(key)
            v.append(val)
        v = np.array(v)
        b = ax.bar(range(len(c)), v[:, 0], label="irrigation", alpha=1.0,)
        for i, k in enumerate(c):
            #if i != 0:#if "base" not in k.lower():
             #   b[i].set_color('#ff7f0e')  # assume two scens only
            #else:
             #   b[i].set_color('#1f77b4')
            b[i].set_color(colors[i])
        b = ax.bar(range(len(c)), v[:, 1], bottom=v[:, 0], label="spray", alpha=0.3)
        for i, k in enumerate(c):
            #if i != 0:#if "base" not in k.lower():
             #   b[i].set_color('#ff7f0e')  # assume two scens only
            #else:
             #   b[i].set_color('#1f77b4')
            b[i].set_color(colors[i])
        b = ax.bar(range(len(c)), v[:, 2], bottom=v[:, 0], label="tip", alpha=0.7)
        for i, k in enumerate(c):
            #if i != 0:#if "base" not in k.lower():
             #   b[i].set_color('#ff7f0e')  # assume two scens only
            #else:
             #   b[i].set_color('#1f77b4')
            b[i].set_color(colors[i])
        ax.legend()
        plt.xticks(range(len(c)), c)
        ax.set_xticklabels([mapper[x].split("_")[-1] for x in c])
        plt.ylabel("Cost Contributions ($)")
        plt.show()
        #plt.savefig(os.path.join("plots", "cost_contribs.pdf"))
        #plt.close()

    return fig, ax

def lai_to_canopy_disease_mgmt(lai_irr_scen, d, mapper):
    # variables
    labour_rate = 25.0  # $/hr
    fuel_cost = 1.2  # $/L  # diesel
    fuel_effic = 0.167 * 60  # L/hr; assuming 60 horsepower tractor from https://agriculture.coerco.com.au/agriculture-blog/how-to-document-diesel-fuel-consumption-on-your-farm and https://winesvinesanalytics.com/features/article/161455/Product-Focus-Narrow-Vineyard-Tractors
    block_rowwise_dist = 89 * 18  # m  # TODO: pass as arg

    spray_cost = lai_to_spray_cost(lai_irr_scen, d, mapper)#block_rowwise_dist, fuel_effic, fuel_cost, labour_rate)
    tip_cost = lai_to_tip_cost(lai_irr_scen, d, mapper)#block_rowwise_dist, fuel_effic, fuel_cost, labour_rate)

    return spray_cost, tip_cost

def lai_to_spray_cost(lai_irr_scen, d, mapper):#block_rowwise_dist, fuel_effic, fuel_cost, labour_rate):
    #spray_pass_speed = 8.0  # km/h
    #hr_per_spray = ((block_rowwise_dist + (0.1 * block_rowwise_dist)) / 1000) / spray_pass_speed

    # fuel
    #fuel_consump_per_spray = fuel_effic * hr_per_spray
    #fuel_cost_per_spray = fuel_consump_per_spray * fuel_cost
    # labour
    #labour_cost_per_spray = labour_rate * hr_per_spray
    # chemical spray
    # chemical_cost = # per L
    # dilution required - including water cost?
    #chemical_cost_per_spray = 100

    # relate lai to number of sprays - viti domain knowledge required here!
    # TODO: relate lai_irr_scens to nsprays
    nsprays = lai_irr_scen.copy()
    nsprays = {x: d[mapper[x]]['nsprays'] for x, a in nsprays.items()}
    #for k, v in lai_irr_scen.items():
     #   if v.max() > 1.9:
      #      nsprays[k] = 7  # 10
       # else:
        #    nsprays[k] = 7

    # spray cost
    spray_cost = nsprays.copy()
    #sum_costs_per_spray = fuel_consump_per_spray + labour_cost_per_spray + chemical_cost_per_spray
    #sum_costs_per_spray = d[]
    #print("cost per spray: {}".format(sum_costs_per_spray))
    #spray_cost = {x: a * sum_costs_per_spray for (x, a) in spray_cost.items()}
    spray_cost = {x: d[mapper[x]]['costs_per_spray'] for x, a in spray_cost.items()}    

    return spray_cost

def lai_to_tip_cost(lai_irr_scen, d, mapper):#block_rowwise_dist, fuel_effic, fuel_cost, labour_rate):
    #tip_pass_speed = 2.0  # km/h
    #hr_per_tip = ((block_rowwise_dist + (0.1 * block_rowwise_dist)) / 1000) / tip_pass_speed

    # fuel
    #fuel_consump_per_tip = fuel_effic * hr_per_tip
    #fuel_cost_per_tip = fuel_consump_per_tip * fuel_cost
    # labour
    #labour_cost_per_tip = labour_rate * hr_per_tip

    # relate lai to number of tips - viti domain knowledge required here!
    # TODO: relate lai_irr_scens to ntips
    ntips = lai_irr_scen.copy()
    ntips = {x: d[mapper[x]]['ntips'] for x, a in ntips.items()}
    #for k, v in lai_irr_scen.items():
     #   if v.max() > 1.9:
      #      ntips[k] = 5  # 10
       # else:
        #    ntips[k] = 5

    # tip cost
    tip_cost = ntips.copy()
    #sum_costs_per_tip = fuel_consump_per_tip + labour_cost_per_tip
    #print("cost per tip: {}".format(sum_costs_per_tip))
    #tip_cost = {x: a * sum_costs_per_tip for (x, a) in tip_cost.items()}
    tip_cost = {x: d[mapper[x]]['costs_per_tip'] for x, a in tip_cost.items()}   

    return tip_cost


def conceptual_phenol_stage_plot():
    df = pd.read_csv(os.path.join(new_model_ws, vines_out_fname),
                     index_col="DayOfYear")
    ax = plt.subplot()
    ax.scatter(df.index, df.loc[:, "iStage"], marker='o')
    ax.xaxis  #set_major_locator(matplotlib.dates.YearLocator(base=1))

def run_base_model():
    # copy frun script
    shutil.copy(os.path.join(frun_vinelogic),
                os.path.join(new_model_ws, frun_vinelogic))
    # copy base input files
    for f in os.listdir(os.path.join(base_model_ws)):
        shutil.copy(os.path.join(base_model_ws, f), os.path.join(new_model_ws, f))
    # run
    pyemu.os_utils.run("Rscript run_vinelogic.R", cwd=new_model_ws)

def run_model(wd):
    # copy frun script if needed
    if not os.path.exists(os.path.join(wd, frun_vinelogic)):
        shutil.copy(os.path.join(frun_vinelogic),
                    os.path.join(wd, frun_vinelogic))
    # run
    pyemu.os_utils.run("Rscript run_vinelogic.R", cwd=wd)


def ins_from_csv(csv):
    # don't call this on its own
    # TODO: obs group by col functionality into pyemu

    # scrape csv and only include relevant cols  # TODO
    cols = pd.read_csv(csv, nrows=0).columns
    cols = [x for x in cols if x != "DayOfYear"]  # as it includes "-"s

    odf = pyemu.pst_utils.csv_to_ins_file(os.path.join(csv), only_cols=cols)

    return odf

def tpl_from_json(json_fname, tpl_identifier="~"):

    # scrape json
    js = pd.read_json(json_fname, convert_dates=True)
    skip_pars = []
    parval1 = {}
    new_js = js.copy()
    for i, v in js.items():
        if i == "FileDescription" or "Latitude" in i or "Longitude" in i \
                or "VINEMSTART" in i or "NumberZones" in i \
                or "Tbase" in i or "NIRR" in i \
                or (json_fname == "BerryCultivar.json" and "SetDD" in i):
            continue
        if isinstance(v.Value, str) or isinstance(v.Value, list) or \
                v.Value is np.nan or v.Value == vinelogic_nan or v.Value == 0.0:  # assume we cannot treat "normal-ly"? TODO: for now...
            #if "-" in v.Value or "/" in v.Value or "date" in v.Units.lower():
            skip_pars.append(i)  # TODO: convert Date to DOY - if not captured in "summary variables"
        else:
            parval1[i] = v.Value
            new_js[i].Value = " {0}   {1}   {0}"\
                .format(tpl_identifier, i)

    # write
    new_js.to_json(json_fname + ".tpl", indent=2)

    # add file tpl header
    # not fast but these aren't big and only once required
    with open(json_fname + ".tpl", 'r') as f:
        lines = f.readlines()
    with open(json_fname + ".tpl", 'w') as f:
        f.write("ptf {}\n".format(tpl_identifier))
        f.writelines(lines)

    strip_json_value_fields(json_fname + ".tpl")  # barf

    return parval1, skip_pars

def strip_json_value_fields(json_fname):
    #
    # absolute barf
    with open(json_fname, 'r') as f:
        lines = f.readlines()
    with open(json_fname, 'w') as f:
        for l in lines:
            if '"Value":" ~' in l:
                lsplit = l.split('"')
                lsplit[1] = '"' + lsplit[1] + '"'
                lagain = ''.join(lsplit)  # worst function ever!
                f.write(lagain)
            else:
                f.write(l)

def json_leading_decimals():
    # https://stackoverflow.com/questions/26898431/are-decimals-without-leading-zeros-valid-json
    # see `run_vinelogic.R` work-around  # TODO: for now. Change pestpp for writing json par files
    pass

def write_time_series_mult_par_tpl(fname):

    file = os.path.join(fname)
    df = pd.read_csv(file)
    ts_pars = [x.replace(" ", "_") + "_mult" for i, x in enumerate(df.columns)
               if i != 0 and "Unnamed" not in x]
    # just simple multiplier for each time series variable for now...
    tpl_fname = os.path.join(fname.replace(".csv", "_mult.csv.tpl"))
    # write tpl
    with open(tpl_fname, 'w') as f:
        f.write("ptf ~\n")
        f.write(",".join([x for x in df.columns if "Unnamed" not in x]) + "\n")
        for ts in ts_pars:
            f.write(",~  {}  ~".format(ts.replace(" ", "_").lower()))
        f.write("\n")
    # write csv
    with open(tpl_fname.replace(".tpl", ""), 'w') as f:
        f.write(",".join([x for x in df.columns if "Unnamed" not in x]) + "\n")
        for ts in ts_pars:
            f.write(",1.000000")
        f.write("\n")

    pv1 = {i.replace(" ", "_").lower(): 1.0 for i in ts_pars}

    return tpl_fname, pv1

def apply_time_series_mult_pars(fnames):
    # for `run_vinelogic.R`
    # writing R in python #barf
    if len([x for x in os.listdir() if frun_vinelogic in x]) > 0:
        frun = os.path.join(frun_vinelogic)
    else:
        frun = os.path.join("..", frun_vinelogic)
    with open(os.path.join(frun), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(frun_vinelogic), 'w') as f:
        flag = False
        for i, l in enumerate(lines):
            if flag is True:
                f.write("\n#time series mult pars..\n")
                for fname in fnames:
                    base_fname = fname.split(".csv")[0] + "_base.csv"
                    if not os.path.exists(os.path.join(base_fname)):
                        shutil.copy2(fname, base_fname)
                    mult_fname = base_fname.replace("base", "mult")

                    f.write("base<-read.csv(file.path(i_dir,'{}',fsep = .Platform$file.sep))\n".format(base_fname))
                    f.write("mult<-read.csv(file.path(i_dir,'{}',fsep = .Platform$file.sep))\n".format(mult_fname))
                    f.write("for (i in colnames(mult)){\n")
                    f.write(" if (i != 'Date'){\n")
                    f.write("  base[[i]] = base[[i]] * mult[[i]]\n")
                    f.write(" }\n")
                    f.write("}\n")
                    f.write("write.csv(base,file.path(getwd(),'{}',fsep = .Platform$file.sep),row.names=FALSE)\n"
                            .format(fname))
                f.write("#..time series mult pars\n")
                flag = False
            elif l.startswith("i_dir<-"):
                flag = True
            f.write(l)


def define_par_bounds(par, pv1_dict):
    # naive approach here as a first pass
    factor = 1.075
    pub = {i: v * factor for (i, v) in pv1_dict.items()}
    plb = {i: v * (1 / factor) for (i, v) in pv1_dict.items()}
    par.loc[:, "parubnd"] = par.index.map(pub)
    par.loc[:, "parlbnd"] = par.index.map(plb)

    # par_bounds_map = {}  # TODO: seek input from Cas, Vinay, CSIRO, etc here.

def setup_pst_interface(all_vinelogic_input_par_packages=True, all_vinelogic_output_files=True,
                        weather_mult_pars=True, irr_mult_pars=False, run=True,
                        seq_da=False, obs_times=True):

    os.chdir(os.path.join(new_model_ws))

    # generate ins files for model outputs
    if isinstance(all_vinelogic_output_files, str):
        all_vinelogic_output_files = [all_vinelogic_output_files]
    if isinstance(all_vinelogic_output_files, list):
        out_file = [os.path.join(x) for x in all_vinelogic_output_files]
    elif all_vinelogic_output_files is True:
        out_file = [os.path.join(vines_out_fname), os.path.join(vines_summ_fname)]
    else:
        raise Exception("illegal entry for `all_vinelogic_output_files`")
    odfs = []
    for csv in out_file:
        odf = ins_from_csv(csv)
        odfs.append(odf)
    odfs = pd.concat(odfs)
    ins_file = [x + ".ins" for x in out_file]

    # TODO: ensure obs nme lengths in ins < 20 char
    #obs.loc[:, "obsnme"] = obs.obsnme.apply(lambda x: x[:20])

    # generate tpl files for model parameters
    pv1 = {}
    skip_pars = {}
    if isinstance(all_vinelogic_input_par_packages, list):
        json_files = all_vinelogic_input_par_packages
    elif isinstance(all_vinelogic_input_par_packages, str):
        json_files = [all_vinelogic_input_par_packages]
    elif all_vinelogic_input_par_packages is True:
        json_files = [x for x in os.listdir() if x.endswith(".json")]
    else:
        raise Exception("illegal entry for `all_vinelogic_input_par_packages`")
    json_files = [x for x in json_files if "Rule" not in x and "Initial" not in x]  # no adjustable pars
    for js in json_files:
        parval1, skip = tpl_from_json(json_fname=js)
        pv1.update(parval1)
        skip_pars[js] = skip
    in_file = [os.path.join(x) for x in json_files]
    tpl_file = [x + ".tpl" for x in in_file]

    # TODO: centralize json and csv par setup into funcs
    mult_par_fnames = []
    if weather_mult_pars is True:
        # and ``external factor'' (e.g., weather) parameters
        tpl_f, iv = write_time_series_mult_par_tpl(weather_input_fname)
        pv1.update(iv)
        in_file.append(tpl_f.replace(".tpl", ""))
        tpl_file.append(tpl_f)
        mult_par_fnames.append(weather_input_fname)

    clean_irrig_csv()
    if irr_mult_pars is True:
        tpl_f, iv = write_time_series_mult_par_tpl(irrig_input_fname)
        pv1.update(iv)
        in_file.append(tpl_f.replace(".tpl", ""))
        tpl_file.append(tpl_f)
        mult_par_fnames.append(irrig_input_fname)

    if weather_mult_pars is True or irr_mult_pars is True:
        apply_time_series_mult_pars(mult_par_fnames)

    # build pst
    pst = pyemu.helpers.pst_from_io_files(tpl_file, in_file, ins_file, out_file)

    par = pst.parameter_data
    par.loc[:, "partrans"] = "none"
    #par.loc[:, "parchglim"] = "relative"
    # initial vals
    pv1 = {i.lower(): v for (i, v) in pv1.items()}
    par.loc[:, "parval1"] = par.index.map(pv1)
    # bounds
    define_par_bounds(par=par, pv1_dict=pv1)
    # groups
    par.loc[:, "pargp"] = par.parnme.apply(lambda x: x)

    # observations
    obs = pst.observation_data
    # obsval
    obs.loc[odfs.index, "obsval"] = odfs.obsval
    # groups
    obs.loc[:, "obgnme"] = obs.obsnme.apply(lambda x: '_'.join(x.split('_')[1:]))
    # weights
    obs.loc[:, "weight"] = 0.0
    #for obs_type in obs_types:
     #   obs.loc[obs.obgnme == obs_type, "weight"] = 1.0
    for obs_type, obs_weight in obs_dict.items():
        if obs_times is True:
            print("no obs_times arg given; assuming obs at all times")
            on = [x for x in obs.obsnme if obs_type in x]  # all
        elif isinstance(obs_times, int):  # temp resolution
            on = [x for x in obs.obsnme if obs_type in x][::obs_times]
        elif isinstance(obs_times, list):  # specific dates
            on = [x for x in obs.obsnme if obs_type in x and int(x.split("_")[0]) in obs_sim_times]
        obs.loc[on, "weight"] = obs_weight  # TODO: spec noise level  # this corresponds to an (expected) noise standard deviation of X...
    #obs.loc[obs.obgnme == "calflux", "weight"] = 0.01  # corresponding to an (expected) noise standard deviation of 100 m^3/d...

    if all_vinelogic_output_files is True or all_vinelogic_output_files == vines_summ_fname:
        lump_summ_var_id = [x for x in obs.obsnme if "doy" in x] + ["1_harvest_at_target_brix", "1_leaf_fall_by_thermal_time"]
        obs.loc[lump_summ_var_id, "obgnme"] = "istage"

    # shorten obgnme
    #obs.loc[:, "obgnme"] = obs.obgnme.apply(lambda x: x[:11])

    pst.control_data.noptmax = 0
    pst.model_command = ["Rscript run_vinelogic.R"]

    pst.write(pst.filename)

    os.chdir(os.path.join(".."))

    # tests

    # (purposeful) hack - to test ins
    # copy orig out_file (used to gen ins) after model run
    #shutil.copy2(os.path.join(new_model_ws, vines_out_fname),
     #            os.path.join(new_model_ws, vines_out_fname+".cp"))
    #with open(os.path.join(new_model_ws, frun_vinelogic), 'a') as f:  # writing R in python # barf
     #   f.write("\nfile.rename('vines_output.csv.cp','vines_output.csv')\n")

    if run is True:
        run_pest_fwd(pst, new_model_ws)

    # prep for DA
    if seq_da is True:  # pestpp-da
        setup_ppda(pst, new_model_ws)

    return pst

def clean_irrig_csv():
    df = pd.read_csv(os.path.join(irrig_input_fname))
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df = df.dropna(axis=0)
    df.to_csv(irrig_input_fname, index=False)


def run_pest_fwd(pst, cwd):
    # standard gauss-marq-leven pestpp-glm from https://github.com/usgs/pestpp
    # ... (my fork thereof)
    exe = "pestpp-glm.exe"
    shutil.copy(os.path.join("exe", exe), os.path.join(cwd))
    pyemu.os_utils.run("{} {}".format(exe, pst.filename), cwd=cwd)

def invest_gsa(pcf, cwd, plot_fname="gsa.pdf"):

    exe = "pestpp-sen.exe"
    #if not os.path.exists(os.path.join(cwd, exe)):
    shutil.copy2(os.path.join("exe", exe), os.path.join(cwd))

    pst = pyemu.Pst(os.path.join(cwd, pcf))
    pst.pestpp_options["tie_by_group"] = True  # as per pestpp manual
    pst.write(os.path.join(cwd, pcf))

    if num_workers > 1:
        pyemu.helpers.start_workers(worker_dir=cwd, exe_rel_path=exe,
                                    pst_rel_path=pcf, num_workers=num_workers,
                                    master_dir="master", worker_root=".")
        # post process
        mio = os.path.join("master", pcf.replace(".pst", ".mio"))  # per obs
        gsa_results = pd.read_csv(mio, index_col="parameter_name")
    else:
        pyemu.os_utils.run("{} {}".format(exe, pcf), cwd=cwd)
        # post process
        # msn = os.path.join(cwd, pcf.replace(".pst", ".msn"))  # per phi
        # per obs
        mio = os.path.join(cwd, pcf.replace(".pst", ".mio"))  # per obs
        gsa_results = pd.read_csv(mio, index_col="parameter_name")

    # par type summ
    par_d, par_l = write_json_par_summ(cwd, pst)
    weather_pars = (list(set(pst.par_names) - set([x.lower() for x in par_l])))
    par_d["Weather"] = weather_pars
    par_l += weather_pars

    # plot
    # summ
    obs_of_interest = {x: x for x in pst.obs_names if "leaf" not in x.lower() and "harvest" not in x.lower()}
    # output time series
    obs_of_interest = {x: x for x in pst.obs_names if x.startswith("833_")}
    #obs_of_interest = {"1_doy03": "DayOfYearFirstFlower", "1_doy05": "DayOfYearVerasion", "1_ffindex": "FruitfulnessIndex", "1_yield": "YieldEndSeason"}
    ncols = 2
    nrows = int(len(obs_of_interest) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    for i, ax in enumerate(axs.reshape(-1)):
        sen = gsa_results.loc[gsa_results.observation_name == list(obs_of_interest.keys())[i], :]
        abs_mean_sens = sen[['sen_mean_abs']]
        # abs_mean_sens["pc_sen_mean_abs"] = (abs_mean_sens.loc[:] / abs_mean_sens.sum()) * 100
        # abs_mean_sens["pc_sen_mean_abs"].sort_values(ascending=False).plot(kind="bar", ax=ax)
        #np.log10(abs_mean_sens['sen_mean_abs']).sort_values(ascending=False).plot(kind="bar", ax=ax)
        np.log10(abs_mean_sens['sen_mean_abs'].reindex(par_l)).plot(kind="bar", ax=ax)
        ax.set_ylabel("{}".format(list(obs_of_interest.items())[i][1]), ) #rotation=0)
        ax.set_ylim([-5, 5])
        #ax.yaxis.set_label_coords(-0.1, 1.02)
        # move ylabel to left

    #ax.set_xlabel("model parameter")
    ax.set_xlabel("", labelpad=0)
    plt.text(0.5, -17, "Cultivar\nparameters", ha='center')
    plt.text(24.5, -17, "Hydrology, Rootstock\nand Soil parameters", ha='center')
    plt.text(43.0, -17, "Weather\nparameters", ha='center')
    plt.text(35.0, -17, "Vineyard\nparameters", ha='center')
    plt.text(9.5, -17, "Control\nvariables", ha='center')
    plt.text(16.5, -17, "More cultivar\nparameters", ha='center')
    #plt.title('log abs mean global sensitivity', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", plot_fname))
    #plt.text(x, y, "cultivar")
    #plt.text(x, y, "cultivar")


def invest_jco(pcf, cwd, plot_fname="jco.pdf", run=True):
    # again see https://github.com/usgs/pestpp - or my fork thereof

    exe = "pestpp-glm.exe"
    #if not os.path.exists(os.path.join(cwd, exe)):
    shutil.copy2(os.path.join("exe", exe), os.path.join(cwd))

    pst = pyemu.Pst(os.path.join(cwd, pcf))
    pst.control_data.noptmax = -1
    if exe == "pest.exe":
        pst.pestpp_options["der_forgive"] = True
    pst.write(os.path.join(cwd, pcf))

    '''
    # hack when using work-around
    if exe == "pest.exe":  # i.e. not pestpp
        with open(os.path.join(cwd, pcf), 'r+') as f:
            t = f.readlines()
        with open(os.path.join(cwd, pcf), 'w+') as f:
            for i, l in enumerate(t):
                if i == 5:  # line containing `derforgive`
                    l = l.replace("noderforgive", "derforgive")
                    f.write(l)
                else:
                    f.write(l)'''

    if run:
        pyemu.os_utils.run("{} {}".format(exe, pcf), cwd=cwd)

    if exe == "pest.exe":
        jco = pyemu.Jco.from_binary(os.path.join(
            cwd, pcf.replace(".pst", ".jco"))).to_dataframe()
    else:
        jco = pyemu.Jco.from_binary(os.path.join(
            cwd, pcf.replace(".pst", ".jcb"))).to_dataframe()

    # plot
    fig, ax = plt.subplots()
    m = np.log10(np.abs(jco.loc[[x for x in jco.index if x.startswith("1_") or x.startswith("801_")]]))
    im = ax.imshow(m.values)
    ax.set_xticks(np.arange(len(m.columns)))
    ax.set_xticklabels(m.columns)
    ax.set_xlabel("model parameters")
    ax.set_yticks(np.arange(len(m.index)))
    ax.set_yticklabels(m.index)
    ax.set_ylabel("summary output variables")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    cbar = plt.colorbar(im, shrink=0.4)
    cbar.set_label(r"$\log(|\frac{\Delta o_{i}}{\Delta p_{j}}|)$")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", plot_fname))
    plt.close()

    # yield finite diff sens plot
    fig, ax = plt.subplots()
    np.log10(np.abs(jco.loc["1_yield"]).sort_values(ascending=False)).plot(kind="bar", ax=ax)
    ax.set_ylabel("log abs sensitivity of Yield (kg/ha)")
    ax.set_xlabel("model parameter")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "sens_Yield_"+plot_fname))

def invest_data_worth(pcf, cwd, potential_only=False, pot_start_end_monitor_date=["2000-09-15", "2001-05-01"],
                      pot_monitor_freq=7, niter_next_best=10):

    # pst
    pst = pyemu.Pst(os.path.join(cwd, pcf))
    #pst.write_par_summary_table(filename="none")
    if potential_only:
        pst.observation_data.loc[:, "weight"] = 0.0

    # point to pre-calculated jco
    jco = os.path.join(cwd, pcf.replace(".pst", ".jcb"))
    # already plotted jco above

    # plot prior/post cov from pst/pestpp-glm
    #cov = pyemu.Cov.from_binary(os.path.join(m_d, "prior_cov.jcb")).to_dataframe()
    #cov = cov.loc[pst.adj_par_names, pst.adj_par_names]
    #cov = pyemu.Cov.from_dataframe(cov)
    #x = cov.x.copy()
    #x[x < 1e-7] = np.nan
    #c = plt.imshow(x)
    #plt.colorbar()
    cov = pyemu.Cov.from_parameter_data(pst)

    # the magic happens here
    sc = pyemu.Schur(pst=pst, jco=jco, parcov=cov, forecasts=forecasts)
    # do this even if potential_only is True (nnz_obs = 0)

    if potential_only is False:
        # part of post cov...
        # sc.posterior_parameter.to_dataframe().sort_index(axis=1).iloc[100:105:, 100:105]

        #x = sc.posterior_parameter.x.copy()
        #x[x < 1e-7] = np.nan
        #c = plt.imshow(x)
        #plt.colorbar(c)

        par_sum = sc.get_parameter_summary().sort_values("percent_reduction", ascending=False)
        par_sum.loc[par_sum.index[:25], "percent_reduction"].plot(kind="bar", color="turquoise")

        # forecast analysis
        df = sc.get_forecast_summary()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax = df["percent_reduction"].plot(kind='bar', ax=ax, grid=True)
        ax.set_ylabel("percent uncertainy\nreduction from calibration")
        ax.set_xlabel("forecast")
        plt.tight_layout()

        # parameter contribution to forecasts
        par_contrib = sc.get_par_group_contribution()
        base = par_contrib.loc["base", :]
        par_contrib = 100.0 * (base - par_contrib) / par_contrib
        par_contrib.sort_index()
        for forecast in par_contrib.columns:
            fore_df = par_contrib.loc[:, forecast].copy()
            fore_df.sort_values(inplace=True, ascending=False)
            ax = fore_df.iloc[:10].plot(kind="bar", color="b")
            ax.set_title(forecast)
            ax.set_ylabel("percent variance reduction")
            plt.show()

        # now, data worth!
        # first with existing obs
        dw_rm = sc.get_removed_obs_importance()
        base = dw_rm.loc["base", :]
        dw_rm = 100 * (dw_rm - base) / dw_rm
        for forecast in dw_rm.columns:
            fore_df = dw_rm.loc[:, forecast].copy()
            fore_df.sort_values(inplace=True, ascending=False)
            ax = fore_df.iloc[:10].plot(kind="bar", color="b")
            ax.set_title(forecast)
            ax.set_ylabel("percent variance increase")
            plt.show()

    # potential obs
    z_obs = pst.observation_data.loc[(pst.observation_data.weight == 0), "obsnme"].tolist()
    z_obs = [x for x in z_obs if x not in forecasts]  # less our forecasts

    pot_obs_types = ["lai"]
    start_end_monitor = date_to_sim_day(dates=pot_start_end_monitor_date)
    new_obs = []
    for pot in pot_obs_types:
        new = [x for x in z_obs if pot in x][::pot_monitor_freq]
        new_obs += new
    new_obs = [x for x in new_obs if int(x.split("_")[0]) > start_end_monitor[0]
               and int(x.split("_")[0]) < start_end_monitor[1]]
    print("number of new potential observations considered: {}".format(len(new_obs)))
    df_worth_new = sc.get_added_obs_importance(obslist_dict=new_obs, base_obslist=sc.pst.nnz_obs_names,
                                               reset_zero_weight=True)
    # some processing
    df_new_base = df_worth_new.loc["base", :].copy()  # "base" row
    df_new_imax = df_worth_new.apply(lambda x: df_new_base - x, axis=1).idxmax()  # obs with largest unc red for each pred
    df_new_worth = 100.0 * (df_worth_new.apply(lambda x: df_new_base - x, axis=1) / df_new_base)  # normalizing like above
    # plot prep
    idx = df_new_worth.index.copy()
    df_new_worth.loc[:, "idx"] = idx
    df_new_worth.loc["base", "idx"] = "0_lai"
    df_new_worth.loc[:, "Date"] = df_new_worth.idx.apply(lambda x: sim_day_to_date([int(x.split("_")[0])])[0])
    df_new_worth.set_index("Date", inplace=True)
    # end hack
    df_new_worth = df_new_worth.append(df_new_worth.iloc[-1, :])
    as_list = df_new_worth.index.to_list()
    as_list[-1] += start_date + timedelta(900) - as_list[-1]
    df_new_worth.index = as_list

    # ``next best`` analysis - recursive
    from datetime import datetime
    start = datetime.now()
    next_most_df = sc.next_most_important_added_obs(forecast=forecasts[0], niter=niter_next_best,
                                                    obslist_dict=dict(zip(new_obs, new_obs)),
                                                    base_obslist=sc.pst.nnz_obs_names,
                                                    reset_zero_weight=True)
    print("recursive 'next best' analysis took...", datetime.now() - start)
    idx = next_most_df.index.copy()
    next_most_df.loc[:, "idx"] = idx
    next_most_df.loc[:, "Date"] = next_most_df.idx.apply(lambda x: sim_day_to_date([int(x.split("_")[0])])[0])
    next_most_df.set_index("Date", inplace=True)

    # plotting
    # run base model for base state var time series
    pst.control_data.noptmax = 0
    pst.write(os.path.join(cwd, pcf))
    pyemu.os_utils.run("pestpp-glm.exe {}".format(pcf), cwd=cwd)
    df = pd.read_csv(os.path.join(cwd, vines_out_fname), index_col="DayOfYear")
    df.loc[:, "Date"] = df.index
    df.loc[:, "Date"] = df.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df.set_index("Date", inplace=True)
    # TODO: generalize for multiple potential obs types... loop over types here
    #  potentially for multiple forecast types too
    fig, axs = plt.subplots(2, 1, sharex=True)
    for i, ax in enumerate(axs):
        if i == 0:  # top
            df.loc[:, "LAI"].plot(ax=ax, color="k")
            ax.set_ylabel("LAI", fontsize=10)
            xlim = ax.get_xlim()
        else:
            df_new_worth[forecasts[0]].resample('1D').ffill().plot(ax=ax, color='turquoise')
            ax.set_ylabel("% reduction $\sigma^{2}_{Yield}$", fontsize=10)
            ax.set_xlim(xlim)
            ax.set_xlabel("Date", fontsize=10)
            # next best markers
            #ax.bar(x=next_most_df.index, height=next_most_df["unc_reduce_iter_base"],
            #       width=timedelta(pot_monitor_freq))
            #for ii, (i, v) in enumerate(next_most_df.iterrows()):
             #   ax.scatter(x=i, y=v["unc_reduce_iter_base"], c="r")
              #  ax.text(x=i, y=v["unc_reduce_iter_base"] + 1, s=ii, fontsize=10)
            #best_lst = [sim_day_to_date([int(x.split("_")[0])]) for x in next_most_df.best_obs.to_list()]
            #for i, v in enumerate(best_lst):
             #   ax.scatter(x=v, y=df_new_worth.loc[v, forecasts[0]], c="r")
            #for i, txt in enumerate(best_lst):
             #   plt.text(x=datetime.strftime(txt[0] + timedelta(14), '%Y-%m-%d'),
              #           y=df_new_worth.loc[txt, forecasts[0]][0] + 0.1, s=i, fontsize=10, ha='center', va='top')
    plt.savefig(os.path.join("plots", "dw_w_next_best.pdf"))


def write_json_par_summ(cwd, pst):
    # to establish approp par unc definition via collab with Cas, Vinay
    dfs, d, l = [], {}, []
    jsons = [x for x in os.listdir(os.path.join(cwd)) if x.endswith(".json.tpl")]
    for jsf in jsons:
        jsf = jsf.replace(".tpl", "")
        df = pd.read_json(os.path.join(cwd, jsf))
        if "Units" not in df.index:  # inconsistent index # TODO: address in vines too
            df.index = [x.replace("Unit", "Units") for x in df.index.to_list()]
        dfs.append(df)
        pars = [x.lower() for x in df.columns if "FileDescription" not in x]
        pars = list(set([x.lower() for x in pst.par_names]).intersection(pars))  # only those in pst
        d[jsf.replace(".json", "")] = pars
        l += pars
    df = pd.concat(dfs, axis=1, join='inner').T
    df.to_csv(os.path.join(cwd, "json_par_summ.csv"))

    return d, l

def setup_ppda(pst, cwd, use_ies=False, num_reals=30):
    # from existing pst obj or new pst?
    # check use new pcf fmt
    #if pst._version_:  # TODO: no such attribute currently..
    pst.pestpp_options["da_use_ies"] = use_ies
    #pst.pestpp_options["ies_parameter_ensemble"] = "kh_ensemble0.csv"
    #pst.pestpp_options["da_parameter_ensemble"] = "kh_ensemble0.csv"
    pst.pestpp_options["ies_num_reals"] = num_reals  # TODO: change to da_num_reals
    pst.control_data.noptmax = 1
    # add `cycle` to interface files
    # write temporal tpl and ins files
    # add `cycle` to par and obs dfs
    pst.write(os.path.join(cwd, pst.filename), version=2)
    # treat ``cycle'' as key word - at least for pyemu setup_ppda()
    # obs_data
    obs = pst.observation_data
    obs.loc[:, "cycle"] = obs.obsnme.apply(lambda x: int(x.split('_')[0]) - 1)
    #cycle_map = {}
    #for i, t in enumerate(sorted([int(x.split("_")[0]) for x in pst.nnz_obs_names])):
     #   o = [x for x in pst.nnz_obs_names if str(t) in x]
      #  obs.loc[o, "cycle"] = int(i)  # TODO: double-check idxing
       # cycle_map[o[0]] = int(i)  # will come in handy
    # TODO: strip leading idx from summary var obsnme
    # obs.loc[obs.cycle == -1, "type"] = "stat"  # TODO: needed?
    # obs.loc[obs.cycle != -1, "type"] = "obs"  # TODO: needed?
    # par_data
    par = pst.parameter_data
    par.loc[:, "cycle"] = -1  #.obsnme.apply(lambda x: x.split('_')[0])
    pst.write(os.path.join(cwd, pst.filename), version=2)
    # insfile_data and tplfile_data - do after write pst for now # TODO: construct df for insfile_data
    # split up ins and obs file into single-time pairs
    out_w_cycles, out_wo_cycles = [vines_out_fname], [vines_summ_fname]  # TODO: automated?
    # TODO: func for csv time series based ins. Args for "include header in each ins/out pair" and "temporal resolution"
    for pair in out_w_cycles:
        pst.instruction_files.remove(pair + ".ins")
        pst.output_files.remove(pair)
    for ofn in out_w_cycles:
        with open(os.path.join(cwd, ofn), 'r+') as out_f:
            out_f_lines = out_f.readlines()
            out_hdr = out_f_lines[0]
            out_f_lines = out_f_lines[1:]
        with open(os.path.join(cwd, ofn + ".ins"), 'r+') as ins_f:
            ins_f_lines = ins_f.readlines()  # skip overall hdr
            ins_identifier = ins_f_lines[0]
            ins_f_lines = ins_f_lines[2:]
        for t, line in enumerate(out_f_lines):  # TODO: assumption here is in sequence and temporal resolution correct - read in as df with datetime index
            fname = ofn.rsplit('.', 1)[0] + "_{}.".format(t) + ofn.rsplit('.', 1)[-1]
            with open(os.path.join(cwd, fname), 'w+') as n_out:
                n_out.write(out_hdr)
                n_out.write(line)
            pst.output_files.append(fname)
        for t, line in enumerate(ins_f_lines):  # TODO: assumption here is in sequence and temporal resolution correct - read in as df with datetime index
            fname = ofn.rsplit('.', 1)[0] + "_{}.".format(t) + ofn.rsplit('.', 1)[-1] + '.ins'
            with open(os.path.join(cwd, fname), 'w+') as n_ins:
                n_ins.write("{}".format(ins_identifier))
                n_ins.write("l1\n")  # hdr in each file..
                n_ins.write(line)
            pst.instruction_files.append(fname)
    # frun script
    frun = "forward_run.py"
    shutil.copy2(os.path.join("ext_funcs.py"), os.path.join(cwd))
    with open(os.path.join(cwd, frun), 'w+') as f:
        f.write("import os\nimport pyemu\nimport ext_funcs\n")  # TODO: put these funcs in pyEMU!
        # `conda init --all` to automatically activate base conda env on shell entrance
        f.write("# prep tpl and par file data\n")
        f.write("pyemu.os_utils.run('{0}')\n".format(pst.model_command.pop()))
        f.write("# prep ins and out file data\n")
        f.write("ext_funcs.prep_out_files_seq_da({})\n".format(out_w_cycles))
    pst.model_command = ['python {}'.format(frun)]
    pst.write(os.path.join(cwd, pst.filename), version=2)
    # insfile_data
    insfd = pd.read_csv(os.path.join(cwd, "{}.insfile_data.csv".format(pst.filename.split('.pst')[0])))
    insfd.loc[:, "cycle"] = insfd.model_file.apply(lambda x: x.split('.csv')[0].split('_')[-1])
    #for i in cycle_map.items():
     #   target_t = int(i[0].split("_")[0])  #TODO: check idxg
      #  insfd.loc[insfd.model_file.str.contains(str(target_t)), "cycle"] = int(i[1])
    insfd.loc[insfd['model_file'].apply(lambda x: x in out_wo_cycles), "cycle"] = -1
    insfd.to_csv(os.path.join(cwd, "{}.insfile_data.csv".format(pst.filename.split('.pst')[0])), index=False)
    # tplfile_data
    tplfd = pd.read_csv(os.path.join(cwd, "{}.tplfile_data.csv".format(pst.filename.split('.pst')[0])))
    #tplfd.loc[:, "cycle"] = tplfd.model_file.apply(lambda x: x.split('.csv')[0].split('_')[-1])
    #tplfd.loc[tplfd['model_file'].apply(lambda x: x in out_wo_cycles), "cycle"] = -1
    tplfd.loc[:, "cycle"] = -1
    tplfd.to_csv(os.path.join(cwd, "{}.tplfile_data.csv".format(pst.filename.split('.pst')[0])), index=False)
    # checking for presence of ``cycle'' entry in par_data, obs_data, insfile_data etc. done at pestpp end
    exe = "pestpp-da.exe"
    shutil.copy2(os.path.join("exe", exe), os.path.join(cwd))
    #pyemu.os_utils.run("{} {}".format(exe, pst.filename), cwd=cwd)
    # TODO: keep ins/obs 1-indexed if model is - only cycle has to be 0-indexed...
    m_d = "master_{0}pars_{1}reals_{2}obs_pestpp-da".format(pst.npar_adj, num_reals, pst.nobs)
    pyemu.helpers.start_workers(cwd, exe, pst.filename, num_workers=num_workers,
                                master_dir=m_d, worker_root=".")
    return pst

def perturb_irrig_data(pcf, cwd, irr_mult=1.0):
    pst = pyemu.Pst(os.path.join(cwd, pcf))
    irr_mult_parnme = [x for x in pst.par_names if "irr" in x and "mult" in x]
    pst.parameter_data.loc[irr_mult_parnme, "parval1"] = irr_mult
    pst.write(os.path.join(cwd, pcf))
    exe = "pestpp-glm.exe"
    pyemu.os_utils.run("{} {}".format(exe, pcf), cwd=cwd)

def plot_helper():
    # TODO: generalize!
    # bar plots for summ vars
    m = pd.read_csv(os.path.join("medium_irrig", vines_summ_fname))
    m.index = ["medium_irrig"]
    l = pd.read_csv(os.path.join("low_irrig", vines_summ_fname))
    l.index = ["low_irrig"]
    h = pd.read_csv(os.path.join("high_irrig", vines_summ_fname))
    h.index = ["high_irrig"]
    df = pd.concat((l, m, h))
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    ps = ["DOY", "Yield"]
    fig, axs = plt.subplots(nrows=len(ps), ncols=1, sharex=False)
    for i, ax in enumerate(axs.reshape(-1)):
        if "DOY" in ps[i]:
            np.log10(df.loc[:, [x for x in df.columns if ps[i] in x]]).T.plot(kind='bar', ax=axs[i])
            xtl = ax.get_xticklabels()
            ax.set_xticklabels(xtl, rotation=0)
            ax.set_ylabel("DayOfYear (phenological transition)")
        else:
            df.loc[:, [x for x in df.columns if ps[i] in x]].plot(kind='bar', ax=axs[i])
            xtl = ax.get_xticklabels()
            ax.set_xticklabels(xtl, rotation=0)
            ax.set_ylabel("Yield [kg/ha]")
    plt.savefig(os.path.join("plots", "summ_vars_irrig_scens.pdf"))
    plt.close()

def run_ies(pcf, cwd, noptmax, num_reals=30):
    exe = "pestpp-ies.exe"
    shutil.copy2(os.path.join("exe", exe), os.path.join(cwd))

    pst = pyemu.Pst(os.path.join(cwd, pcf))
    pst.control_data.noptmax = noptmax
    pst.pestpp_options = {}
    pst.pestpp_options["ies_use_prior_scaling"] = "False"
    pst.pestpp_options["ies_use_approx"] = "True"
    pst.pestpp_options["ies_num_reals"] = num_reals
    #pst.pestpp_options["ies_lambda_mults"] = [0.1, 1.0, 10.0]
    #ies_subset_size = int(num_workers / (len(pst.pestpp_options.get("ies_lambda_mults")) *
     #                                    len(pst.pestpp_options.get("lambda_scale_fac"))))
    #pst.pestpp_options["ies_subset_size"] = ies_subset_size
    #pst.pestpp_options["parcov_filename"] = "pest.cov.jcb"  # TODO: draw using pyemu to capture correlation
    #pst.pestpp_options["ies_parameter_ensemble"] = par_en
    #pst.pestpp_options["ies_restart_observation_ensemble"] = obs_en
    #pst.pestpp_options["ies_bad_phi"] = 1.0e+30

    pst.pestpp_options["forecasts"] = forecasts

    #pst.pestpp_options["ies_localizer"] = "loc.mat"  # localize  # TODO: temporal localization

    pst.write(os.path.join(cwd, pcf))

    m_d = "master_{0}pars_{1}reals_{2}obs_{3}_pestpp-ies".format(pst.npar_adj, num_reals, pst.nobs, noptmax)
    pyemu.helpers.start_workers(cwd, exe, pcf, num_workers=num_workers,
                                master_dir=m_d, worker_root=".")

    return m_d

def plot_ies_results(pcf, m_d, final_it=None):

    pst = pyemu.Pst(os.path.join(m_d, pcf))
    par = pst.parameter_data
    pdict = par.groupby("pargp").groups
    obs = pst.observation_data

    bins = 15

    # parameter ens
    pe_pr = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".0.par.csv")), index_col=0)
    if pst.control_data.noptmax < 0:  # just prior MC ies run
        pyemu.plot_utils.ensemble_helper({"0.5": pe_pr}, plot_cols=pdict, bins=bins,
                                         filename="prior_par_pdfs.pdf")
    else:  # da
        if final_it is not None:
            noptmax = final_it
        else:
            noptmax = max([int(x.split(".")[-3]) for x in os.listdir(os.path.join(m_d))
                           if x.endswith(".par.csv")])
        pe_pt = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".{0}.par.csv"
                                                          .format(noptmax))), index_col=0)
        pyemu.plot_utils.ensemble_helper({"0.5": pe_pr, "b": pe_pt}, plot_cols=pdict, bins=bins,
                                         filename="prior_posterior_par_pdfs.pdf")

    # observation ens
    oe_pr = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".0.obs.csv")), index_col=0)
    forecasts = pst.pestpp_options["forecasts"].split(",")
    if pst.control_data.noptmax < 0:  # just prior MC ies run
        for forecast in forecasts:
            ax = plt.subplot(111)
            oe_pr.loc[:, forecast].hist(ax=ax, color="0.5", alpha=0.5, label='prior')
            #ax.plot([obs.loc[forecast, "obsval"], obs.loc[forecast, "obsval"]], ax.get_ylim(), "r", label='truth')
            ax.set_title(forecast)
            ax.legend(loc='upper right')
            plt.savefig("prior_{0}_{1}reals.pdf".format(forecast, m_d.split("reals")[0].split("_")[-1]))
            plt.close()
    else:
        oe_pt = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".{0}.obs.csv"
                                                          .format(noptmax))), index_col=0)
        for forecast in forecasts:
            ax = plt.subplot(111)
            oe_pr.loc[:, forecast].hist(ax=ax, color="0.5", alpha=0.5, label='prior')
            oe_pt.loc[:, forecast].hist(ax=ax, color="b", alpha=0.5, label='posterior')
            #ax.plot([obs.loc[forecast, "obsval"], obs.loc[forecast, "obsval"]], ax.get_ylim(), "r", label='truth')
            ax.set_title(forecast)
            ax.legend(loc='upper right')
            plt.savefig("prior_posterior_{0}_{1}reals.pdf".format(forecast, m_d.split("reals")[0].split("_")[-1]))
            plt.close()

    # phi and misfit
    #pst.phi
    #plt.figure()
    #pst.plot(kind='phi_pie');
    #print('Here are the non-zero weighted observation contributions to phi')
    #plot_phi_progress(m_d)

    #figs = pst.plot(kind="1to1");
    #pst.res.loc[pst.nnz_obs_names, :]
    #plt.show()

    # state variable time series
    plot_ts_en(pcf=pcf, m_d=m_d, final_it=final_it)

def setup_truth_model(pcf, cwd, obsnme_for_truth_real_select, percentile_for_truth_real_select,
                      num_reals=1000, add_noise_to_obs=True, add_model_error=False, obs_times=True):

    # run large prior MC sweep
    m_d = run_ies(pcf=pcf, cwd=cwd, noptmax=-1, num_reals=num_reals)
    plot_ies_results(pcf=pcf, m_d=m_d)

    # and process
    obs_df = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".0.obs.csv")), index_col=0)
    print('number of realization in the ensemble before dropping: ' + str(obs_df.shape[0]))
    #obs_df = obs_df.loc[obs_df.failed_flag == 0, :]  # TODO: no failed flag with most recent V?
    #print('number of realization in the ensemble **after** dropping: ' + str(obs_df.shape[0]))

    # candidate truth realization index
    sorted_vals = obs_df.loc[:, obsnme_for_truth_real_select].sort_values(ascending=True)
    idx = sorted_vals.index[int(percentile_for_truth_real_select / 100 * num_reals)]
    print("selected {0}th prior realization sorted (index: {1}) with respect to {2} as truth model.."
          .format(percentile_for_truth_real_select, idx, obsnme_for_truth_real_select))
    #obs_df.loc[idx, pst.nnz_obs_names]  # check

    # replace obsval with the outputs of the selected realizations in obs en
    pst = pyemu.Pst(os.path.join(cwd, pcf))
    obs = pst.observation_data
    obs.loc[:, "obsval"] = obs_df.loc[idx, pst.obs_names]
    #assert np.isclose(obs.loc[obsnme_for_truth_real_select, "obsval"],
     #                 obs_df.loc[idx, obsnme_for_truth_real_select], atol=1e-3)  # TODO: assert obs_df.loc[:, "1_yield"].sort_values()[int(.2 * len(obs_df.loc[:, "1_yield"]) + 2)]

    # weights - turn some obs on
    obs.loc[:, "weight"] = 0.0
    for obs_type, obs_weight in obs_dict.items():
        if obs_times is True:
            print("no obs_times arg given; assuming obs at all times")
            on = [x for x in obs.obsnme if obs_type in x]  # all
        elif isinstance(obs_times, str):  # temp resolution
            on = [x for x in obs.obsnme if obs_type in x][::obs_times]
        elif isinstance(obs_times, list):  # specific dates
            on = [x for x in obs.obsnme if obs_type in x and int(x.split("_")[0]) in obs_sim_times]
        obs.loc[on, "weight"] = obs_weight  # TODO: spec noise level  # this corresponds to an (expected) noise standard deviation of X...
    #obs.loc[obs.obgnme == "calflux", "weight"] = 0.01  # corresponding to an (expected) noise standard deviation of 100 m^3/d...

    if add_noise_to_obs:
        np.random.seed(seed=0)
        snd = np.random.randn(pst.nnz_obs)
        noise = snd * 1. / obs.loc[pst.nnz_obs_names, "weight"]
        pst.observation_data.loc[noise.index, "obsval"] += noise

    if add_model_error:
        pass

    # now test..
    # phi
    # TODO: see `plot_phi_progress()`
    pcf_truth_obs = pcf.replace(".pst", "_fwd_w_truth_obs.pst")
    pst.control_data.noptmax = 0  # TODO: check
    pst.write(os.path.join(m_d, pcf_truth_obs))
    pyemu.os_utils.run("pestpp-ies.exe {0}".format(pcf_truth_obs), cwd=m_d)

    pst = pyemu.Pst(os.path.join(m_d, pcf_truth_obs))
    #print(pst.phi)
    plt.figure()
    figs = pst.plot(kind='phi_pie')
    plt.savefig("phi_pie.pdf")
    plt.close()
    figs = pst.plot(kind="1to1")
    #pst.res.loc[pst.nnz_obs_names, :]
    plt.savefig("1to1.pdf")
    plt.close()

    # now run model with truth real pars - where misfit should be zero!
    par_df = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".0.par.csv")), index_col=0)
    pst.parameter_data.loc[:, "parval1"] = par_df.loc[str(idx), pst.par_names]
    pcf_truth_obs_pars = pcf.replace(".pst", "_fwd_w_truth_obs_and_truth_pars.pst")
    pst.write(os.path.join(m_d, pcf_truth_obs_pars))
    pyemu.os_utils.run("pestpp-ies.exe {0}".format(pcf_truth_obs_pars), cwd=m_d)
    pst = pyemu.Pst(os.path.join(m_d, pcf_truth_obs_pars))
    #assert np.isclose(pst.phi, 0.0, atol=1e-3)
    #pst.res.loc[pst.nnz_obs_names, :]  # TODO: assert based on yield too

    # copying/renaming in cwd for next step - DA(!)
    shutil.copy2(os.path.join(cwd, pcf), os.path.join(cwd, pcf.replace(".pst", "_base_model_obs.pst")))
    shutil.copy2(os.path.join(m_d, pcf_truth_obs), os.path.join(cwd, pcf))

    # how does existing obs en compare to truth real
    # forecasts
    obs = pst.observation_data  # note pst with truth pars and obs here!
    forecasts = pst.pestpp_options["forecasts"].split(",")
    plt.figure()
    for forecast in forecasts:
        ax = plt.subplot(111)
        obs_df.loc[:, forecast].hist(ax=ax, color="0.5", alpha=0.5)
        ax.plot([obs.loc[forecast, "obsval"], obs.loc[forecast, "obsval"]], ax.get_ylim(), "r")
        ax.set_title(forecast)
        plt.savefig("prior_{}_compare_to_truth.pdf".format(forecast))
        plt.close()

    '''# and obs
    for oname in pst.nnz_obs_names:
        ax = plt.subplot(111)
        obs_df.loc[:, oname].hist(ax=ax, color="0.5", alpha=0.5)
        ax.plot([obs.loc[oname, "obsval"], obs.loc[oname, "obsval"]], ax.get_ylim(), "r")
        ax.set_title(oname)
        plt.savefig("prior_{}_compare_to_truth.pdf".format(oname))
        plt.close()'''

    # pipe to truth model obs to csv
    out_obs = obs.loc[obs.obsnme.apply(lambda x: x in pst.nnz_obs_names or x in forecasts), :]
    out_obs.to_csv(os.path.join("truth_model_obs_data.csv"))


def plot_phi_progress(m_d):
    phi = pd.read_csv(os.path.join(m_d, "freyberg_ies.phi.actual.csv"), index_col=0)
    phi.index = phi.total_runs
    phi.iloc[:, 6:].apply(np.log10).plot(legend=False, lw=0.5, color='k')
    plt.ylabel('log \$Phi$')
    plt.figure()
    phi.iloc[-1, 6:].hist()
    plt.title('Final $\Phi$ Distribution')

def sim_day_to_date(sim_days):
    dates = [start_date + timedelta(x) for x in sim_days]
    return dates

def date_to_sim_day(dates):
    sim_days = [datetime.strptime("{}".format(x), '%Y-%m-%d') - start_date for x in dates]
    sim_days = [(x + timedelta(1)).days for x in sim_days]  # 1-indexing
    return sim_days

def weather_input_file_checks_and_fills():
    data_file = os.path.join("_data", "bom_loxton_weath_for_vinelogic.csv")  # see README in base dir
    df = pd.read_csv(data_file, index_col=0)
    cols = ["SRAD", "TMAX", "TMIN", "ET"]
    for col in cols:
        df[col].interpolate(inplace=True)
    df["RAIN"].fillna(value=0, inplace=True)
    df.to_csv(os.path.join("_data", "MILD.csv"))

def build_and_run_model(wd):
    # copy base input files
    for f in os.listdir(os.path.join(base_model_ws)):
        shutil.copy(os.path.join(base_model_ws, f), os.path.join(new_model_ws, f))
    # and already prepared case study-specific files (from wd)
    for f in os.listdir(os.path.join(wd)):
        if not os.path.isdir(os.path.join(wd, f)):
            shutil.copy(os.path.join(wd, f), os.path.join(new_model_ws, f))
    # run
    pyemu.os_utils.run("Rscript run_vinelogic.R", cwd=new_model_ws)

def plot_ts_en(pcf, m_d, final_it):

    pst = pyemu.Pst(os.path.join(m_d, pcf))
    obs = pst.observation_data
    oe_pr = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".0.obs.csv")), index_col=0)
    if pst.control_data.noptmax > 0:
        if final_it is not None:
            noptmax = final_it
        else:
            noptmax = max([int(x.split(".")[-3]) for x in os.listdir(os.path.join(m_d))
                       if x.endswith(".par.csv")])
        oe_pt = pd.read_csv(os.path.join(m_d, pcf.replace(".pst", ".{0}.obs.csv"
                                                          .format(noptmax))), index_col=0)
    #sim_day_to_date()
    ts_vars = states
    for ts_var in ts_vars:
        # TODO: use subplots()
        vs = [x for x in oe_pr.columns if ts_var.lower() in x]
        ax = plt.subplot(111)
        idx = [int(x.split("_")[0]) for x in vs]  # TODO: pad obsnmes with zeros
        idx.sort()
        i = ["{0}_{1}".format(x, ts_var.lower()) for x in idx]
        oe_pr.loc[:, vs].T.reindex(i).plot(ax=ax, color="0.5", alpha=0.5)
        #obs.loc[vs, "obsval"].reindex(i).plot(ax=ax, color='r', label='truth')
        handles, labels = plt.gca().get_legend_handles_labels()
        d = dict(zip(labels, handles))
        d = {k.replace("base", "prior"): v for k, v in d.items() if "base" in k
             or "truth" in k}  # always assume "base" real works...
        ax.legend(d.values(), d.keys(), loc='upper left')
        if pst.control_data.noptmax > 0:
            oe_pt.loc[:, vs].T.reindex(i).plot(ax=ax, color="b", alpha=0.5)
            handles, labels = plt.gca().get_legend_handles_labels()
            d2 = dict(zip(labels, handles))
            d2 = {k.replace("{0}".format(labels[-1]), "posterior"): v for k, v in d2.items()
                  if "{0}".format(labels[-1]) in k}  # always assume "base" real works...
            d2.update(d)
            ax.legend(d2.values(), d2.keys(), loc='upper left')
        #else:
         #   ax.legend(d.values(), d.keys(), loc='upper left')
        ax.set_title(ts_var)
        labels = [(start_date + timedelta(days=int(x.get_text().split("_")[0]))).date()
                  for x in ax.get_xticklabels() if x.get_text() != '']
        ax.set_xticklabels(labels)
        ax.set_xlabel("Date")
        if pst.control_data.noptmax > 0:
            plt.savefig("prior_posterior_ts_{0}_{1}reals.pdf".format(ts_var, m_d.split("reals")[0].split("_")[-1]))
        else:
            plt.savefig("prior_ts_{0}_{1}reals.pdf".format(ts_var, m_d.split("reals")[0].split("_")[-1]))
        plt.close()


def run_scen(ws):
    #print(scen_d.keys())
    #for ws in scen_d.keys():
    #scen_ws = os.path.join(ws)
    #print("running {} model".format(scen_ws))
    run_model(wd=os.path.join(ws))  # TODO: until R env ported...
    #return list(scen_d.keys())

    #scens = ["irrig_base"]
    #if "low" in irrig_scen:
    #    irrig_scen = str("irrig_" + irrig_scen)
    #    scens = [irrig_scen] + scens
    #else:
    #    scens.append("irrig_" + irrig_scen)
    #for scen in scens:
    #    scen_ws = os.path.join(scen)
    #    #print("running {} model".format(scen_ws))
    #    #run_model(wd=scen_ws)  # TODO: until R env ported...
    #    #ts_plot_helper(cwd=scen_ws, fname="state_ts_{}.pdf".format(scen))
    #return scens

    #infeas, phi = run_pestpp_opt(const_dict,risk,extra_sw_consts)
    #fig, ax = plot_scenario_dv(infeas,extra_sw_consts,risk)
    #ax.set_title(" $\phi$ (\\$/yr): {0:<15.3G}, risk: {1:<15.3G}".format(phi,risk))
    #return fig, ax

def water_balance_components_plot(cwds, d, plot_type):
    states = ["rain", "irrigation", "runoff", "evap", "Tru"]
    dfs = {}
    for i, scen_ws in enumerate(cwds):
        df = pd.read_csv(os.path.join(scen_ws, vines_out_fname),
                         index_col="DayOfYear")
        dates = pd.date_range(start_date, periods=daily_time_steps)
        df.index = dates

        summ_df = pd.read_csv(os.path.join(scen_ws, vines_summ_fname))
        bb = summ_df.loc[:, "DOY02"][0]
        hv = summ_df.loc[:, "DOY06"][0]
        bb = doy_to_date(bb, (start_date + timedelta(365)).year).strftime("%d %b %Y")
        hv = doy_to_date(hv, (start_date + timedelta(365*3)).year).strftime("%d %b %Y")
        
        #df = df.loc[datetime.strftime(start_date + timedelta(365 + (365 / 2) + 61), '%Y-%m-%d'):
         #           datetime.strftime(start_date + timedelta(365*2 + (365/2)), '%Y-%m-%d'), :] #timedelta(900), '%Y-%m-%d'), :]
        df = df.loc[bb:hv, :]

        df = df.loc[:, states]
        for mm_col in ["rain", "irrigation"]:
            df.loc[:, mm_col] = df[mm_col].apply(lambda x: x / 10.0)
            dfs[scen_ws] = df.sum()

    dfs = pd.DataFrame(dfs)
    print(dfs)

    if plot_type == 'bar':
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        for out in ["runoff", "evap", "Tru"]:
            print(dfs[out, :], [x for x in dfs.columns])
            #dfs[out, [x for x in dfs.columns]].apply(lambda x: x * -1)
        print(dfs)

        x = np.arange(len(dfs[scen_ws].index))
        print(x)
        w = 0.25

        for i, col in enumerate(dfs.columns):
            print(i, col)
            b = ax.bar(x[i], s.values, w, label=col)

    elif plot_type == 'pie':
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        axs = np.array(axs)
        print(dfs)
        dfs.apply(lambda x: x / dfs.sum())
        #for i, col in enumerate(dfs.columns):
        for i, ax in enumarate(axs.reshape(-1)):
            p = ax.pie(s, labels=s.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
    
    return fig, ax


def plot_scen(scens, plot, scen_d):

    if plot == "irrigationtimeseries":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="irrigation", d=scen_d)
    elif plot == "irrigationtotal":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="irrigation", d=scen_d, total=True)
    #elif plot == "irrigationtotal":
     #   _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="irrigationtotal", d=scen_d)
    elif plot == "irrigationcost":
        q_irr_scen, _, _ = ts_compare_irrig_plot(cwds=scens, which="irrigation", d=scen_d, show_plot=False)
        _, (fig, ax) = irrig_compare(q_irr_scen, d=scen_d, mapper=scens)
    elif plot == "harvestyield":
        #q_irr_scen, lai_irr_scen, _ = ts_compare_irrig_plot(cwds=scens, which="irrigation", show_plot=False)
        _, (fig, ax) = yield_revenue_compare(cwds=scens, which="yield", d=scen_d)
    elif plot == "harvestrevenue":
        _, (fig, ax) = yield_revenue_compare(cwds=scens, which="revenue", d=scen_d)
    elif plot == "laitimeseries":
        q_irr_scen, lai_irr_scen, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="lai", d=scen_d)
    elif plot == "fruittimeseries":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="fruit", d=scen_d)
    elif plot == "brixtimeseries":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="Brix", d=scen_d)

    elif plot == "fruitsinkts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="FruitSink", d=scen_d)
    elif plot == "cpoolts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="Cpool", d=scen_d)
    elif plot == "swstressts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="soil_water_stress1", d=scen_d)
    #elif plot == "supplydemand":
     #   _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="supply_demand", d=scen_d)
    elif plot == "VineEop":
         _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="VineEop", d=scen_d)
    elif plot == "Tru":
         _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="Tru", d=scen_d)

    elif plot == "infiltrationts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="infiltration", d=scen_d)
    elif plot == "evaporationts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="evap", d=scen_d)
    elif plot == "drainagets":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="drainage", d=scen_d)
    elif plot == "watertablets":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="water_table", d=scen_d)
    elif plot == "soilmoisturets":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="soil_water", d=scen_d)
    elif plot == "athetats":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="ATheta", d=scen_d)
    elif plot == "swbts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="soil_water_balance", d=scen_d)
    elif plot == "tsw1ts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="total_soil_water1", d=scen_d)
    #elif plot == "tsw2ts":
     #   _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="total_soil_water2", d=scen_d)
    elif plot == "tswtopts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="TswTop", d=scen_d)
    elif plot == "wet1ts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="Wet1", d=scen_d)
    elif plot == "rootuptakets":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="root_uptake", d=scen_d)
    elif plot == "raints":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="rain", d=scen_d)
    elif plot == "raintotal":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="rain", d=scen_d, total=True)
    elif plot == "pondts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="ponding", d=scen_d)
    elif plot == "runoffts":
        _, _, (fig, ax) = ts_compare_irrig_plot(cwds=scens, which="runoff", d=scen_d)

    elif plot == "swbpie":
        fig, ax = water_balance_components_plot(cwds=scens, d=scen_d, plot_type='pie')
    elif plot == "swbbar":
        fig, ax = water_balance_components_plot(cwds=scens, d=scen_d, plot_type='bar')
    
    elif plot == "costcontributions" or plot == "grossmargin":
        q_irr_scen, lai_irr_scen, _ = ts_compare_irrig_plot(cwds=scens, which="irrigation", d=scen_d, show_plot=False)
        dolla_irr_scen, _ = irrig_compare(q_irr_scen, d=scen_d, mapper=scens, show_plot=False)
        revenue, _ = yield_revenue_compare(cwds=scens, which="revenue", d=scen_d, show_plot=False)
        _, lai_irr_scen, _ = ts_compare_irrig_plot(cwds=scens, which="lai", d=scen_d, show_plot=False)
        spray_cost, tip_cost = lai_to_canopy_disease_mgmt(lai_irr_scen, d=scen_d, mapper=scens)
        if plot == "grossmargin":
            fig, ax = gross_margin(dolla_irr_scen, revenue, which="gross_margin", d=scen_d, mapper=scens,
                include_disease_mgmt=True, include_canopy_mgmt=True, spray_cost=spray_cost, tip_cost=tip_cost)
        else:
            fig, ax = gross_margin(dolla_irr_scen, revenue, which="cost_contribs", d=scen_d, mapper=scens,
                include_disease_mgmt=True, include_canopy_mgmt=True, spray_cost=spray_cost, tip_cost=tip_cost)
    elif "date" in plot:
        fig, ax = plot_phenol_keydate(cwds=scens, which=plot, d=scen_d)
    elif plot == "underdev":
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.text(0.5, 0.5, "UNDER CONSTRUCTION", color="red", ha='center', va='center', fontsize=24)
        #plt.close()

    return fig, ax

def plot_phenol_keydate(cwds, which, d):
    if not isinstance(cwds, list):  # dict
        dd = cwds.copy()
        cwds = list(cwds.keys())
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    text_height, text_color = [0.9, 0.7, 0.5], [colors[x] for x in range(3)] #['#1f77b4', '#ff7f0e', '#2ca02c']
    #if "base" in cwds[1]:
     #   cwds.reverse()
    for i, scen_ws in enumerate(cwds):
        df = pd.read_csv(os.path.join(scen_ws, vines_summ_fname))
        if "bb" in which:
            plot = "bud burst"
            kd = df.loc[:, "DOY02"][0]
        elif "ff" in which:
            plot = "first flower"
            kd = df.loc[:, "DOY03"][0]
        elif "vs" in which:
            plot = "veraison"
            kd = df.loc[:, "DOY05"][0]
        elif "hv" in which:
            plot = "harvest"
            kd = df.loc[:, "DOY06"][0]
        ax.text(0.5, text_height[i], "Date of {0} ({1}):\n {2}".format(plot.title(), dd[scen_ws].split("_")[-1], doy_to_date(kd, (start_date + timedelta(365)).year).strftime("%d %b %Y")), 
                ha='center', va='center', fontsize=24, color=text_color[i])
        ax.axis('off')
    return fig, ax

def doy_to_date(day, year):
    date = datetime.strptime('{0} {1}'.format(year, day), '%Y %j')
    return date


if __name__ == "__main__":

    #### experimental set up ####
    all_jsons = [x for x in os.listdir(os.path.join(base_model_ws)) if x.endswith(".json")]
    all_vinelogic_input_par_packages = [x for x in all_jsons if "BerryCultivar" not in x and "control" not in x]  #True
    all_vinelogic_output_files = True  #vines_summ_fname  #vines_out_fname
    weather_mult_pars = False
    irr_mult_pars = False  # TODO: use RuleBasedIrrigationData only!
    #### experimental set up ####

    # vineLOGIC checks
    run_base_model()
    #ts_plot_helper(fname="base_state_time_series.pdf")
    #conceptual_phenol_stage_plot()

    # LRC app file setup
    weather_input_file_checks_and_fills()
    study_site_ws = "_lrc_block47"
    build_and_run_model(wd=study_site_ws)

    # interim pred and advice dash workflow
    # TODO: wrap all this in a wrapper func or in notebook (e.g., "include canopy mgmt component?")
    '''scens = ["low_irrig", "high_irrig"]
    for scen in scens:
        scen_ws = os.path.join(scen)
        run_model(wd=scen_ws)
        #ts_plot_helper(cwd=scen_ws, fname="state_ts_{}.pdf".format(scen))
    q_irr_scen, lai_irr_scen = ts_compare_irrig_plot(cwds=scens, fname="ts_{}.pdf".format(scens))
    grape_revenue = yield_revenue_compare(cwds=scens)
    irrig_cost = irrig_compare(q_irr_scen, percent_entitlement=70)
    spray_cost, tip_cost = lai_to_canopy_disease_mgmt(lai_irr_scen)
    gross_margin(irrig_cost, grape_revenue, include_canopy_mgmt=True,
                 include_disease_mgmt=True, spray_cost=spray_cost, tip_cost=tip_cost)
    '''
    if obs_dates is not None:
        obs_sim_times = date_to_sim_day(dates=obs_dates)
    else:
        obs_sim_times = obs_times  # TODO: clean up

    # build pst interface
    seq_da = False
    setup_pst_interface(all_vinelogic_input_par_packages, all_vinelogic_output_files, weather_mult_pars,
                        irr_mult_pars, seq_da=seq_da, obs_times=obs_sim_times)  # includes run tests

    # produce data (input/output sets for diff irrig scens) for AIML
    '''
    # time series for state vars
    ts_plot_helper(cwd="medium_irrig", fname="state_ts_medium.pdf")
    ts_plot_helper(cwd="low_irrig", fname="state_ts_low.pdf")
    ts_plot_helper(cwd="high_irrig", fname="state_ts_high.pdf")
    # summ vars
    plot_helper()
    if irr_mult_pars is True:
        for irr_mult in [1.0, 0.1, 2.0]:
            perturb_irrig_data(pcf=pst_fname, cwd=new_model_ws, irr_mult=irr_mult)
            ts_plot_helper(fname="soil_water_stress_irrmult{}".format(irr_mult), cwd=new_model_ws)
    '''

    # runs
    # sensitivity analysis
    #if all_vinelogic_output_files is True or all_vinelogic_output_files == vines_summ_fname:
        # TODO: remove this dependency - only here to control plotting
        # TODO: plot coeff of variation
     #   invest_gsa(pcf=pst_fname, cwd=new_model_ws, plot_fname="gsa_{0}_{1}.pdf"
      #             .format(all_vinelogic_input_par_packages, all_vinelogic_output_files))
        #invest_jco(pcf=pst_fname, cwd=new_model_ws, plot_fname="jco{}_dw_prep.pdf",
         #          run=True)
        #invest_data_worth(pcf=pst_fname, cwd=new_model_ws, potential_only=True)

    pcf = pst_fname
    cwd = new_model_ws
    noptmax = -1
    num_reals = 100
    m_d = run_ies(pcf=pcf, cwd=cwd, noptmax=noptmax, num_reals=num_reals)
    plot_ies_results(pcf=pcf, m_d=m_d)

    # DA
    #noptmax = 10
    #num_reals = 30
    #setup_truth_model(pcf=pcf, cwd=cwd, obsnme_for_truth_real_select=forecasts[0],
     #                 percentile_for_truth_real_select=15, add_noise_to_obs=False, num_reals=100)
    #m_d = "master_46pars_30reals_36918obs_10_pestpp-ies"  #run_ies(pcf=pcf, cwd=cwd, noptmax=noptmax, num_reals=num_reals)
    #final_it = 8  #noptmax  #3
    #plot_ies_results(pcf=pcf, m_d=m_d, final_it=final_it)

    # other
    #write_json_par_summ(cwd=new_model_ws)

    # TODO: check discrepancy between outputs with base and base within interface
    # TODO: new control file format to harbour par descriptions
    # TODO: programmatically doc changes to model - see "_base_model_test"
    # TODO: parameterize lists in jsons
    # TODO: more generalize plotting funcs
    # TODO: enforce parameter constraints, e.g., bud burst before veraison, ...
