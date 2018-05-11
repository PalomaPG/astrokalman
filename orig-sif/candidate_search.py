# -*- coding: utf-8 -*-
from RunData import RunData
from FITSHandler import FITSHandler
from Observer import Observer

import sys
#sys.path.insert(0, 'SIF')
#import SIF
from time import time




year = '15'
n_params = 32

RD = RunData(year=year, n_params=n_params)
print 'soaaa %d' % RD.n_params
#RD = SIF.RunData(year=year,only_HiTS_SN=only_HiTS_SN,n_params=n_params,filter_type='MCC')
if RD.n_params > 0:
    print 'apply params'
    RD.apply_params()

print 'fuera del if apply params'
init = time()

FH = FITSHandler(RD)
MJD = FH.MJD

KF,SN = RD.deploy_filter_and_detector(MJD)

if RD.SN_index>=0:
    OB = Observer(len(MJD), new_pos=RD.SN_pos)

# First run: collect candidates

"""for k in FH.data_names.keys():
    print k
    print np.array(FH.data_names[k])
    print ''"""


for o in range(len(MJD)):
    print 'Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD))
    
    ti = time()
    FH.load_fluxes(o)
    print 'Load and photometry time: ' + str(time()-ti)
    
    ti = time()
    KF.update(MJD[o],FH)
    print 'Update filter time: ' + str(time()-ti)
    
    ti = time()
    SN.draw_complying_pixel_groups(o,FH,KF)
    print 'Draw groups time: ' + str(time()-ti)
    
    SN.update_candidates(o)
    
    if RD.SN_index>=0:
        OB.rescue_run_data(o,FH,KF,SN)
    
    print ''
    
# Classify new candidates
SN.check_candidates(RD)

print 'SN ' + ['not found ','found '][RD.SN_found]
print 'Number of unidentified objects: ' + str(RD.NUO)

# Evaluate worth of second run
RD.decide_second_run(OB)
    
# Second run: rescue historic info from candidates, if any
KF, SN = RD.deploy_filter_and_detector(MJD)

OB = Observer(len(MJD))
OB.new_objects_from_CandData(RD.CandData)

for o in range(len(MJD)):
    print 'Beginning with observation: ' + str(o+1).zfill(2) + '/' + str(len(MJD))
    
    ti = time()
    FH.load_fluxes(o)
    print 'Load and photometry time: ' + str(time()-ti)
    
    ti = time()
    KF.update(MJD[o],FH)
    print 'Update filter time: ' + str(time()-ti)
    
    ti = time()
    SN.draw_complying_pixel_groups(o,FH,KF)
    print 'Draw groups time: ' + str(time()-ti)
    
    OB.rescue_run_data(o,FH,KF,SN)
    
    print ''

RD.save_results(OB)

print 'Total time: '+str(time()-init)