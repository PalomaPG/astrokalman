# -*- coding: utf-8 -*-
from RunData import RunData
from FITSHandler import FITSHandler
from Observer import Observer
from resource import getrusage as resource_usage, RUSAGE_SELF

from unix_time import unix_time
from time import time
import os

year = '15'
n_params = 32

RD = RunData(year=year, n_params=n_params)
#RD = SIF.RunData(year=year,only_HiTS_SN=only_HiTS_SN,n_params=n_params,filter_type='MCC')
if RD.n_params > 0:
    RD.apply_params()
    #print('Parameter index: ' + str(RD.this_par).zfill(2))

init = time()

fitshand_time = resource_usage(RUSAGE_SELF).ru_utime
FH = FITSHandler(RD)
fitshand_time = resource_usage(RUSAGE_SELF).ru_utime - fitshand_time
print('Time to load files...:%f' % fitshand_time)
MJD = FH.MJD

KF, SN = RD.deploy_filter_and_detector(MJD)
# Si el SN_index significa que estamos en una SN de HiTS. por tanto preparamos un objeto observer
# para crear estampillas de la  SN de HiTS en la primera corrida
if RD.SN_index >= 0:
    OB = Observer(len(MJD), new_pos=RD.SN_pos)

# First run: collect candidates
# Recorre arreglos de MJDs
load_photometry_time = 0
update_filter_time = 0
draw_groups_time = 0
update_candidates_time = 0
i=0
for o in range(len(MJD)):

    print('Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD)))
    print(FH.data_names['science'][i])
    #ti = time()
    # Que FH tenga listos los flujos para pasarlos a KF
    t = unix_time(FH.load_fluxes, (o,))
    load_photometry_time = load_photometry_time + t['user']
    #print ('Load and photometry time: ' + str(time()-ti))
    
    #ti = time()
    t = unix_time(KF.update, (MJD[o],FH,))
    update_filter_time = update_filter_time + t['user']
    #print ('Update filter time: ' + str(time()-ti))
    
    #ti = time()
    # Detector...  Deteccion de candidatos
    t = unix_time(SN.draw_complying_pixel_groups, (o, FH, KF))
    draw_groups_time = draw_groups_time + t['user']
    #print ('Draw groups time: ' + str(time()-ti))
    t = unix_time(SN.update_candidates, (o,))
    update_candidates_time = t['user'] + update_candidates_time
    i +=1
    if RD.SN_index>=0:
        OB.rescue_run_data(o,FH,KF,SN)

find_sn_time = load_photometry_time+update_candidates_time+draw_groups_time+update_candidates_time

print('Loading photometry: %s\n' % str(load_photometry_time))
print('Updating filter: %s\n' % str(update_filter_time))
print('Drawing groups: %s\n' % str(draw_groups_time))
print('Updating candidates: %s\n' % str(update_candidates_time))
print('Total time first routine: %f\n' % find_sn_time)
# Classify new candidates
SN.check_candidates(RD)

print('SN ' + ['not found ','found '][RD.SN_found])
print('Number of unidentified objects: ' + str(RD.NUO))

# Evaluate worth of second run
RD.decide_second_run(OB)
    
# Second run: rescue historic info from candidates, if any
KF, SN = RD.deploy_filter_and_detector(MJD)

OB = Observer(len(MJD))
OB.new_objects_from_CandData(RD.CandData)

load_photometry_time = 0
update_filter_time = 0
draw_groups_time = 0
rescue_time = 0
for o in range(len(MJD)):
    print('Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD)))
    
    #ti = time()
    t = unix_time(FH.load_fluxes, (o,))
    load_photometry_time = load_photometry_time + t['user']
    #load_photometry_time = (time()-ti) + load_photometry_time
    
    #ti = time()
    t = unix_time(KF.update, (MJD[o],FH))
    update_filter_time = update_filter_time + t['user']
    #update_filter_time = (time()-ti) + update_filter_time
    
    #ti = time()
    t = unix_time(SN.draw_complying_pixel_groups, (o, FH, KF,))
    draw_groups_time = draw_groups_time +t['user']
    #draw_groups_time = (time()-ti) + draw_groups_time

    t= unix_time(OB.rescue_run_data, (o, FH, KF, SN))
    rescue_time = rescue_time + t['user']
    

RD.save_results(OB)

print('Loading photometry: %s\n' % str(load_photometry_time))
print('Updating filter: %s\n' % str(update_filter_time))
print('Drawing groups: %s\n' % str(draw_groups_time))
print('Retrieving data to be plotted: %s\n' % str(rescue_time))

print('Total time: %f' % (load_photometry_time+update_candidates_time+draw_groups_time+rescue_time+ find_sn_time))
