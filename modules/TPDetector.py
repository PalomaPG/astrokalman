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
            obj['pixel_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
            obj['group_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))


            self.obj += [obj]

    def look_cand_data(self, cand_data, pred_state, pred_state_cov, kalman_gain, state, state_cov, time_mjd,
                     flux, var_flux, science, diff,  psf, base_mask, dil_base_mask, pixel_flags, group_flags_map, o):

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
            self.obj[i]['pixel_flags'][o, :] = pixel_flags[a:b, c:d]
            self.obj[i]['group_flags'][o, :] = group_flags_map[a:b, c:d]

    def plot_results(self, objects, semester, field, ccd, plot_path):
        vis = Visualizer()
        for i in range(len(objects)):
            obj = objects[i]
            vis.print_lightcurve(obj=obj, obs_rad=self.obs_rad, figsize1=self.figsize1, figsize2=self.figsize2,
                                 save_filename=('%slc_sem_%s_field_%s_ccd_%s_obj_%d.png' % (plot_path,semester,
                                                                                            field, ccd, i)))
            vis.print_stamps(obj, figsize1=self.figsize1, figsize2=self.figsize2,
                             save_filename=('%sstamps_sem_%s_field_%s_ccd_%s_obj_%d.png' % (plot_path,semester,
                                                                                            field, ccd, i)))
            vis.print_space_states(obj=obj, obs_rad=self.obs_rad, figsize1=self.figsize1, figsize2=self.figsize2,
                                   save_filename=('%sspace_states_sem_%s_field_%s_ccd_%s_obj_%d.png' % (plot_path,
                                                                                                        semester, field,
                                                                                                        ccd, i)))
