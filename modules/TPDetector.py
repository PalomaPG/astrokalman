import glob
import numpy as np

from modules.Visualizer import Visualizer


class TPDetector(object):

    def __init__(self):
        #self.visualizer = Visualizer()
        self.cand_coords = []
        self.cand_info = {}

    def look_candidates(self, results_path, field, ccd, semester='15A'):
        regex_path = ('%ssources_sem_' % results_path) + ('%s_mjd_' % semester)+\
                     ('[0-9]'*5)+'.'+ ('[0-9]' * 2)+\
                     ('_field_%s_ccd_%s.npz' % (field, ccd))
        results_list = sorted(glob.glob(regex_path))
        for result in results_list:
            mjd = float(result.split('_')[5])
            data=np.load(result)
            self.list_candidates(data['cand_mid_coords'], mjd)
            data.close()

        return self.cand_coords

    def list_candidates(self,cand_mid_coords, mjd):
        for coords in cand_mid_coords:
            if len(self.cand_coords) == 0:
                self.cand_coords.append(coords)
                self.cand_info[str(coords)] = {'mjd' : list([mjd])}

            else:
                new_candidate=True
                for cand_coords in self.cand_coords:
                    if np.sqrt((cand_coords[0]-coords[0])^2 + (cand_coords[1]-coords[1])^2) < 4.0:
                        new_candidate=False
                        self.cand_info[str(cand_coords)].append(mjd)
                        break
                if new_candidate:
                    self.cand_coords.append(coords)
                    self.cand_info[str(coords)]= list([mjd])

    def get_plots(self, coords, results_path, field, ccd, semester='15A'):

        vis = Visualizer()
        regex_path = ('%ssources_sem_' % results_path) + ('%s_mjd_' % semester)+\
                     ('[0-9]'*5)+'.'+ ('[0-9]' * 2)+\
                     ('_field_%s_ccd_%s.npz' % (field, ccd))
        results_list = sorted(glob.glob(regex_path))
        vis.set_plot(results_list)



