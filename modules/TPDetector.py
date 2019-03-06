import glob
import numpy as np

from modules.Visualizer import Visualizer


class TPDetector(object):

    def __init__(self,x_margin=55, y_margin=55, n_alerts=2):
        #self.visualizer = Visualizer()
        self.cand_coords = []
        self.cand_info = {}
        self.n_obs = 0
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.n_alerts = n_alerts

    def look_candidates(self, results_path, field, ccd, semester='15A'):
        print('----look candidates-----')
        regex_path = ('%ssources_sem_' % results_path) + ('%s_mjd_' % semester)+\
                     ('[0-9]'*5)+'.'+ ('[0-9]' * 2)+\
                     ('_field_%s_ccd_%s.npz' % (field, ccd))
        results_list = sorted(glob.glob(regex_path))
        i_mjd = 1
        self.idx = 1
        if len(results_list)<self.n_alerts:
            raise Exception('There must be at least %d observations' % self.n_alerts)
        else:
            for result in results_list:
                mjd = float(result.split('_')[4])
                data=np.load(result)
                self.list_candidates(data['cand_mid_coords'], mjd, i_mjd)
                i_mjd = i_mjd+1
                self.n_obs = self.n_obs + 1
                data.close()
        print(self.cand_info)
        self.cand_info = clean_dict(self.cand_info, self.n_alerts)
        print(self.cand_info)

    def list_candidates(self,cand_mid_coords, mjd, i_mjd):

        for coords in cand_mid_coords:
            if len(self.cand_coords) == 0:
                self.cand_coords.append(coords)
                self.cand_info[self.idx] = {'mjd' : [mjd], 'coords' : coords, 'mjd_id' : [i_mjd]}
                self.idx = self.idx + 1

            else:
                new_candidate=True
                for cand_coords in self.cand_coords:
                    if np.sqrt((cand_coords[0]-coords[0])^2 + (cand_coords[1]-coords[1])^2) < 4.0:
                        id = search_id(cand_coords, self.cand_info)
                        self.cand_info[id]['mjd'].append(mjd)
                        self.cand_info[id]['mjd_id'].append(i_mjd)
                        new_candidate=False
                        break
                if new_candidate and coords[0]>self.y_margin and coords[1] >self.x_margin:
                    self.cand_coords.append(coords)
                    self.cand_info[self.idx]= {'mjd' : [mjd], 'coords' : coords, 'mjd_id' : [i_mjd]}
                    self.idx = self.idx + 1

    def get_plots(self, results_path, plots_path, field, ccd, semester='15A'):

        vis = Visualizer(results_path=results_path, plots_path=plots_path)
        regex_path = ('%ssources_sem_' % results_path) + ('%s_mjd_' % semester)+\
                     ('[0-9]'*5)+'.'+ ('[0-9]' * 2)+\
                     ('_field_%s_ccd_%s.npz' % (field, ccd))
        results_list = sorted(glob.glob(regex_path))
        vis.set_plot_states(results_list, self.cand_info)


def search_id(coords, dict_):
    id = -1

    for k,v in dict_.items():

        if np.sqrt((coords[0]-v['coords'][0])^2 + (coords[1]-v['coords'][1])^2) < 4.0:
            id = k
            break

    return id


def clean_dict(dict_, n_seq):
    for k in list(dict_.keys()):
        if len(dict_[k]['mjd']) < n_seq:
            del dict_[k]

    return dict_
