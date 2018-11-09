import glob
import numpy as np


class TPDetector(object):

    def __init__(self):
        self.cand_coords = []

    def look_candidates(self, results_path, field, ccd, semester=None):
        regex_path = ('%ssources_mjd_' % results_path) +('[0-9]'*5)+'.'+ ('[0-9]' * 2)+ ('_field_%s_ccd_%s.npz' % (field, ccd))
        results_list = sorted(glob.glob(regex_path))
        for result in results_list:
            data=np.load(result)
            self.list_candidates(data['cand_mid_coords'])
            data.close()

        print(self.cand_coords)
        print(len(self.cand_coords))

    def list_candidates(self,cand_mid_coords):
        for coords in cand_mid_coords:
            if len(self.cand_coords) == 0:
                self.cand_coords.append(coords)
            else:
                new_candidate=True
                for cand_coords in self.cand_coords:
                    if np.sqrt((cand_coords[0]-coords[0])^2 + (cand_coords[1]-coords[1])^2) < 4.0:
                        new_candidate=False
                        break
                if new_candidate:
                    self.cand_coords.append(coords)



