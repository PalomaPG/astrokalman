# -*- coding: utf-8 -*-
from Observer import Observer
#from SIF import SIF
from glob import glob
import numpy as np

param = 'par-00_'
HiTS = '*'

#results_dir = 'C:/Users/Phuentelemu/Dropbox/Supernovae/MCKF/results/'
results_dir = 'C:/cygwin64/home/Phuentelemu/MCKF/results/'

all_results = glob(results_dir + param + 'HiTS' + HiTS + '*.npz')

for result_name in all_results:
    
    objects = np.load(result_name)['objects']
    MJD = objects[0]['MJD']
    
    new_obs = Observer(len(MJD),figsize1=18,figsize2=9.5)
    new_obs.obj = objects
    
    SN_found = result_name.find('AYE') >= 0
    HiTSSN = result_name[result_name.find('HiTS')+4:result_name.find('HiTS')+6]
    
    filename = 'images/' + param + 'HiTS' + HiTSSN
    
    new_obs.print_lightcurve(save_filename=filename,SN_found=SN_found)
    new_obs.print_stamps(save_filename=filename,SN_found=SN_found)
    
    for obj in objects:
        print str(obj['epochs'])+' '+str(obj['posY'])+'-'+str(obj['posX'])+' '+str(obj['status'])