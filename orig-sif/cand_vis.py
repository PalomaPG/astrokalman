# -*- coding: utf-8 -*-
from Observer import Observer

import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib.pyplot as plt

#results_dir = 'C:/cygwin64/home/Phuentelemu/MCKF/results/'
results_dir = 'C:/Users/pablo/Dropbox/Supernovae/MCKF/best_results/'

save_lightcurve = False
save_stamps = False
save_space_states = False
save_total_comparison = False
save_entropy_comparison = False
save_convex_entropy_comparison = True

# Create images directory
images_dir = 'resultsplots'

params = map(lambda x:str(x).zfill(2),range(2))

init_time = time()

if save_total_comparison:
    total_ss_fig = []
    for ind_p in range(len(params)):
        plt.figure(ind_p,figsize=(18,9.5))
        
        max_x_lim = 10000
        min_x_lim = -500
        max_y_lim = 1500
        min_y_lim = -500
        
        flux_thres = 500
        vel_flux_thres = 150
        vel_satu = 3000
        extreme_vel_thres = vel_flux_thres*(vel_satu-min(max_x_lim,vel_satu))/vel_satu
        
        plt.plot([flux_thres,flux_thres,vel_satu,max_x_lim],[max_y_lim,vel_flux_thres,0,0],'k-',label='Thresholds')
        plt.grid()
        plt.xlim(min_x_lim,max_x_lim)
        plt.ylim(min_y_lim,max_y_lim)
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')

for sn in range(93):
    
    for p in params:
        result = glob(results_dir + '/par-' + p + '_HiTS' + str(sn+1).zfill(2) + '*.npz')
        if len(result)>0:
            result = result[0]
        else:
            continue
        
        objects = np.load(result)['objects']
        new_obs = Observer(objects[0]['MJD'],figsize1=18,figsize2=9.5)
        MJD = objects[0]['MJD']
        
        SN_found = result.find('AYE') >= 0
        NUO_counter = 0
        
                              
        for obj in objects:
            
            filename = '/par-' + p + '_HiTS' + str(sn+1).zfill(2)
                              
            if obj['status'] == -1:
                status = '_NUO-'+str(NUO_counter).zfill(2)
                NUO_counter += 1
                filename = '/FP' + filename + status
            else:
                if SN_found:
                    status = '_SN-AYE'
                    filename = '/TP' + filename + status
                else:
                    status = '_SN-nay'
                    filename = '/FN' + filename + status
            
            pos = '_' + str(obj['posY']).zfill(4) +'-'+ str(obj['posX']).zfill(4)
            
            filename = images_dir + filename + pos
            
            if save_lightcurve:
                new_obs.print_lightcurve(MJD,obj,save_filename=filename,SN_found=SN_found)
            if save_stamps:
                new_obs.print_stamps(MJD,obj,save_filename=filename,SN_found=SN_found)
            if save_space_states:
                new_obs.print_space_states(MJD,obj,save_filename=filename,SN_found=SN_found)
                
            if save_total_comparison:
                new_obs.print_all_space_states(int(p),MJD,obj,sn,obj['status'] == -1,SN_found)
                
            if save_entropy_comparison:
                new_obs.print_all_entropy(int(p),MJD,obj,sn,obj['status'] == -1,SN_found,n_bins=20)
                
            if save_convex_entropy_comparison:
                new_obs.print_all_convex_entropy(int(p),MJD,obj,sn,obj['status'] == -1,SN_found)
            
            
                
                
print 'Time: ' + str(time()-init_time)