# -*- coding: utf-8 -*-
from Observer import Observer
from glob import glob
import numpy as np

param = 'par-00_'
HiTS = '*'

#results_dir = 'C:/Users/Phuentelemu/Dropbox/Supernovae/MCKF/results/'
results_dir = '/home/paloma/Documents/Memoria/Code/sif2/sif/results/'

#all_results = glob(results_dir + param + 'HiTS' + HiTS + '*.npz')
all_results = glob(results_dir + '*.npz')
print(all_results)

for result_name in all_results:
    
    objects = np.load(result_name)['objects']
    MJD = objects[0]['MJD']
    
    new_obs = Observer(len(MJD),figsize1=18,figsize2=9.5)
    new_obs.obj = objects
    
    SN_found = result_name.find('AYE') >= 0
    print(SN_found)
    HiTSSN = result_name[result_name.find('HiTS')+4:result_name.find('HiTS')+6]
    
    filename = '/home/paloma/Documents/Memoria/Code/sif2/sif/results/' + param + 'HiTS' + HiTSSN
    print(filename)
    ''
    print(SN_found)
    new_obs.print_lightcurve(MJD=MJD, obj =objects[0], save_filename=filename, SN_found=SN_found)
    new_obs.print_stamps(MJD=MJD, obj =objects[0], save_filename=filename, SN_found=SN_found)
    
    for obj in objects:
        print(str(obj['epochs'])+' '+str(obj['posY'])+'-'+str(obj['posX'])+' '+str(obj['status']))