import glob
import numpy as np

from modules.Visualizer import Visualizer


class TPDetector(object):

    def __init__(self, num_obs, obs_rad=10, figsize1=12, figsize2=8):
        """

        :param num_obs:
        :param obs_rad:
        :param new_pos:
        :param figsize1:
        :param figsize2:
        """
        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.num_obs = num_obs
        self.obs_rad = obs_rad
        self.obs_diam = self.obs_rad * 2 + 1
        self.obj = []
        #if len(new_pos) >= 2:
        #   self.new_object(new_pos[0], new_pos[1], status=1000)

    def set_space(self, cand_data):

        for i in range(len(cand_data)):
            print('In loop...')
            obj = {
                'posY': cand_data[i]['coords'][0],
                'posX': cand_data[i]['coords'][1],
                'epochs': cand_data[i]['epochs']
            }
            a, b = cand_data[i]['coords'][0] - self.obs_rad, cand_data[i]['coords'][0] + self.obs_rad + 1
            c, d = cand_data[i]['coords'][1] - self.obs_rad, cand_data[i]['coords'][1] + self.obs_rad + 1
            obj['a'] = int(a)
            obj['b'] = int(b)
            obj['c'] = int(c)
            obj['d'] = int(d)

            obj['pred_state'] =np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
            obj['pred_state_cov']=np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
            obj['kalman_gain']=np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
            obj['state']=np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
            obj['state_cov']=np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
            obj['MJD'] =np.zeros(self.num_obs)
            obj['obs_flux'] =np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
            obj['obs_var_flux']=np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
            obj['science']=np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
            obj['dil_base_mask']=np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=bool)
            obj['base_mask']=np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=int)
            obj['psf'] =np.zeros((self.num_obs, 21, 21))
            obj['diff']= np.zeros((self.num_obs, self.obs_diam, self.obs_diam))

            self.obj += [obj]


    def look_cand_data(self, cand_data, pred_state, pred_state_cov, kalman_gain, state, state_cov, time_mjd,
                     flux, var_flux, science, diff,  psf, base_mask, dil_base_mask, o):

        """
        :param canda_data:
        :return:
        """
        for i in range(len(cand_data)):
            a = self.obj[i]['a']
            b = self.obj[i]['b']
            c = self.obj[i]['c']
            d = self.obj[i]['d']

            self.obj[i]['pred_state'][o, :] = pred_state[:, a:b, c:d]
            self.obj[i]['pred_state_cov'][o, :] = pred_state_cov[:, a:b, c:d]
            self.obj[i]['kalman_gain'][o, :] = kalman_gain[:, a:b, c:d]
            self.obj[i]['state'][o, :] = state[:, a:b, c:d]
            self.obj[i]['state_cov'][o, :] = state_cov[:, a:b, c:d]
            self.obj[i]['MJD'][o] = time_mjd
            self.obj[i]['obs_flux'][o, :] = flux[a:b, c:d]
            self.obj[i]['obs_var_flux'][o, :] = var_flux[a:b, c:d]
            self.obj[i]['science'][o, :] = science[a:b, c:d]
            self.obj[i]['diff'][o, :] = diff[a:b, c:d]
            self.obj[i]['psf'][o, :] = psf
            self.obj[i]['base_mask'][o, :] = base_mask[a:b, c:d]
            self.obj[i]['dil_base_mask'][o, :] = dil_base_mask[a:b, c:d]


            #obj['pixel_flags'][n_obs, :] = pixel_flags[a:b, c:d]
            #obj['group_flags'][n_obs, :] = group_flags_map[a:b, c:d] #SND.PGData['group_flags_map'][a:b, c:d]


            #self.set_results(canda_data[i]['coords'][0], canda_data[i]['coords'][1],
            #                epochs=canda_data[i]['epochs'])
            #                #status=canda_data[i]['status'])

        #np.savez(canda_data, objects =self.obj)
"""
    def set_results(self, pos_y, pos_x, epochs, pred_state, pred_state_cov, kalman_gain, state, state_cov, time_mjd,
                     flux, var_flux, science, diff, pixel_flags, psf, base_mask, dil_base_mask,
                     group_flags_map, n_obs):

        for i in range(len(self.obj)):
            a, b = self.obj[i]['posY'] - self.obs_rad, self.obj[i]['posY'] + self.obs_rad + 1
            c, d = self.obj[i]['posX'] - self.obs_rad, self.obj[i]['posX'] + self.obs_rad + 1
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            self.obj[i]['pred_state'][n_obs, :] = pred_state[:, a:b, c:d]
            self.obj[i]['pred_state_cov'][n_obs, :] = pred_state_cov[:, a:b, c:d]
            self.obj[i]['kalman_gain'][n_obs, :] = kalman_gain[:, a:b, c:d]
            self.obj[i]['state'][n_obs, :] = state[:, a:b, c:d]
            self.obj[i]['state_cov'][n_obs, :] = state_cov[:, a:b, c:d]
            self.obj[i]['MJD'][n_obs] = time_mjd
            self.obj[i]['obs_flux'][n_obs, :] = flux[a:b, c:d]
            self.obj[i]['obs_var_flux'][n_obs, :] = var_flux[a:b, c:d]
            self.obj[i]['science'][n_obs, :] = science[a:b, c:d]
            self.obj[i]['diff'][n_obs, :] = diff[a:b, c:d]
            self.obj[i]['pixel_flags'][n_obs, :] = pixel_flags[a:b, c:d]
            self.obj[i]['group_flags'][n_obs, :] = group_flags_map[a:b, c:d] #SND.PGData['group_flags_map'][a:b, c:d]
            self.obj[i]['psf'][n_obs, :] = psf
            self.obj[i]['base_mask'][n_obs, :] = base_mask[a:b, c:d]
            self.obj[i]['dil_base_mask'][n_obs, :] = dil_base_mask[a:b, c:d]
    """
"""
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
"""
"""
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
"""