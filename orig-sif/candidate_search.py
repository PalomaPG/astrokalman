# -*- coding: utf-8 -*-
from RunData import RunData
from FITSHandler import FITSHandler
from Observer import Observer


from time import process_time

year = '15'
n_params = 32

RD = RunData(year=year, n_params=n_params)
#RD = SIF.RunData(year=year,only_HiTS_SN=only_HiTS_SN,n_params=n_params,filter_type='MCC')
if RD.n_params > 0:
    print('apply params')
    RD.apply_params()

init = process_time()

FH = FITSHandler(RD)
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

for o in range(len(MJD)):
    print('Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD)))
    
    ti = process_time()
    # Que FH tenga listos los flujos para pasarlos a KF
    FH.load_fluxes(o)
    #print('Load and photometry time: ' + str(process_time()-ti))
    load_photometry_time = (process_time()-ti) + load_photometry_time


    ti = process_time()
    KF.update(MJD[o],FH)
    #print('Update filter time: ' + str(process_time()-ti))
    update_filter_time = (process_time()-ti) + update_filter_time

    ti = process_time()
    # Detector...  Deteccion de candidatos
    SN.draw_complying_pixel_groups(o, FH, KF)
    #print('Draw groups time: ' + str(process_time()-ti))
    draw_groups_time = (process_time()-ti) + draw_groups_time

    SN.update_candidates(o)
    
    if RD.SN_index>=0:
        OB.rescue_run_data(o,FH,KF,SN)

print('\nTiempo total: %f\n' % (load_photometry_time+update_filter_time+draw_groups_time))
print('Loading photometry: %f\n' % (load_photometry_time))
print('Updating filter: %f\n' % (update_filter_time))
print('Drawing groups: %f\n' % (draw_groups_time))

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

for o in range(len(MJD)):
    print('Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD)))
    
    ti = process_time()
    FH.load_fluxes(o)
    load_photometry_time = (process_time()-ti) + load_photometry_time
    
    ti = process_time()
    KF.update(MJD[o],FH)
    update_filter_time = (process_time()-ti) + update_filter_time
    
    ti = process_time()
    SN.draw_complying_pixel_groups(o, FH, KF)
    draw_groups_time = (process_time()-ti) + draw_groups_time
    
    OB.rescue_run_data(o, FH, KF, SN)
    
print('Tiempo total %f' % (load_photometry_time+update_filter_time+draw_groups_time))

RD.save_results(OB)

print('Total time: '+str(process_time()-init))
print('Loading photometry: %s\n' % str(load_photometry_time))
print('Updating filter: %s\n' % str(update_filter_time))
print('Drawing groups: %s\n' % str(draw_groups_time))