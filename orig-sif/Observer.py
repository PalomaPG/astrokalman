# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np
import matplotlib.pyplot as plt


class Observer(object):

    def __init__(self, num_obs, obs_rad=10, new_pos=[], figsize1=12, figsize2=8):
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
        if len(new_pos) >= 2:
            self.new_object(new_pos[0], new_pos[1], status=1000)

    def new_objects_from_CandData(self, CandData):
        """

        :param CandData:
        :return:
        """
        for i in range(len(CandData)):
            self.new_object(CandData[i]['coords'][0], CandData[i]['coords'][1], epochs=CandData[i]['epochs'],
                            status=CandData[i]['status'])

    def new_object(self, posY, posX, epochs=[-1], status=-1):
        """

        :param posY:
        :param posX:
        :param epochs:
        :param status:
        :return:
        """
        new_obj = {'posY': posY, 'posX': posX, 'epochs': epochs, 'status': status}
        new_obj['pred_state'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['pred_state_cov'] = np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
        new_obj['kalman_gain'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['state'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['state_cov'] = np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
        new_obj['MJD'] = np.zeros(self.num_obs)
        new_obj['obs_flux'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['obs_var_flux'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['science'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['diff'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['pixel_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['group_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['psf'] = np.zeros((self.num_obs, 21, 21))
        new_obj['base_mask'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=int)
        new_obj['dil_base_mask'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=bool)
        self.obj += [new_obj]

    def rescue_run_data(self, o, FH, KF, SND):
        """

        :param o:
        :param FH:
        :param KF:
        :param SND:
        :return:
        """
        for i in range(len(self.obj)):
            a, b = self.obj[i]['posY'] - self.obs_rad, self.obj[i]['posY'] + self.obs_rad + 1
            c, d = self.obj[i]['posX'] - self.obs_rad, self.obj[i]['posX'] + self.obs_rad + 1
            self.obj[i]['pred_state'][o, :] = KF.pred_state[:, a:b, c:d]
            self.obj[i]['pred_state_cov'][o, :] = KF.pred_state_cov[:, a:b, c:d]
            self.obj[i]['kalman_gain'][o, :] = KF.kalman_gain[:, a:b, c:d]
            self.obj[i]['state'][o, :] = KF.state[:, a:b, c:d]
            self.obj[i]['state_cov'][o, :] = KF.state_cov[:, a:b, c:d]
            self.obj[i]['MJD'][o] = KF.time
            self.obj[i]['obs_flux'][o, :] = FH.flux[a:b, c:d]
            self.obj[i]['obs_var_flux'][o, :] = FH.var_flux[a:b, c:d]
            self.obj[i]['science'][o, :] = FH.science[a:b, c:d]
            self.obj[i]['diff'][o, :] = FH.diff[a:b, c:d]
            self.obj[i]['pixel_flags'][o, :] = SND.pixel_flags[a:b, c:d]
            self.obj[i]['group_flags'][o, :] = SND.PGData['group_flags_map'][a:b, c:d]
            self.obj[i]['psf'][o, :] = FH.psf
            self.obj[i]['base_mask'][o, :] = FH.base_mask[a:b, c:d]
            self.obj[i]['dil_base_mask'][o, :] = FH.dil_base_mask[a:b, c:d]

    def print_lightcurve(self, MJD, obj, posY=-1, posX=-1, save_filename='', SN_found=False):
        """

        :param MJD:
        :param obj:
        :param posY:
        :param posX:
        :param save_filename:
        :param SN_found:
        :return:
        """

        num_graphs = 4

        if posY == -1:
            posY = self.obs_rad
        if posX == -1:
            posX = self.obs_rad

        # for i in range(len(self.obj)):
        # obj = self.obj[i]
        # MJD = obj['MJD']

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        ax1 = plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.errorbar(MJD + 0.015, obj['state'][:, 0, posY, posX], yerr=obj['state_cov'][:, 0, posY, posX], fmt='b.-',
                     label='Estimated flux')
        plt.errorbar(MJD - 0.015, obj['pred_state'][:, 0, posY, posX], yerr=obj['pred_state_cov'][:, 0, posY, posX],
                     fmt='g.', label='Predicted flux')
        plt.errorbar(MJD, obj['obs_flux'][:, posY, posX], yerr=obj['obs_var_flux'][:, posY, posX], fmt='r.',
                     label='Observed flux')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([min(obj['state'][:, 0, posY, posX]) - 500, max(obj['state'][:, 0, posY, posX]) + 500])
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0), sharex=ax1)
        plt.errorbar(MJD, obj['state'][:, 1, posY, posX], yerr=obj['state_cov'][:, 2, posY, posX], fmt='b.-',
                     label='Estimated flux velocity')
        plt.errorbar(MJD - 0.03, obj['pred_state'][:, 1, posY, posX], yerr=obj['pred_state_cov'][:, 2, posY, posX],
                     fmt='g.', label='Predicted flux velocity')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (2, 0), sharex=ax1)
        plt.plot(MJD, obj['pixel_flags'][:, posY, posX], '.-', label='Pixel flags')
        plt.plot(MJD, obj['group_flags'][:, posY, posX], '.-', label='Pixel Group flags')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (3, 0), sharex=ax1)
        plt.plot(MJD - 0.011, obj['pred_state_cov'][:, 0, posY, posX], 'y.', label='Pred Flux Variance')
        plt.plot(MJD - 0.01, obj['state_cov'][:, 0, posY, posX], 'y+', label='Flux Variance')
        plt.plot(MJD - 0.001, obj['pred_state_cov'][:, 1, posY, posX], 'b.', label='Pred Flux-Velo Variance')
        plt.plot(MJD + 0.00, obj['state_cov'][:, 1, posY, posX], 'b+', label='Flux-Velo Variance')
        plt.plot(MJD + 0.009, obj['pred_state_cov'][:, 2, posY, posX], 'g.', label='Pred Velo Variance')
        plt.plot(MJD + 0.01, obj['state_cov'][:, 2, posY, posX], 'g+', label='Velo Variance')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([0, 200])

        plt.xlabel('MJD [days]')

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_lightcurves', bbox_inches='tight')
        plt.close(this_fig)

    def stack_stamps(self, stamps, MJD, max_value=10000):
        """

        :param stamps:
        :param MJD:
        :param max_value:
        :return:
        """
        stack = stamps[0, :]
        prev_time = MJD[0]
        stamps_diam = stamps.shape[1]
        for i in range(1, stamps.shape[0]):
            stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            if MJD[i] - prev_time > 0.5:
                stack = np.hstack((stack, -max_value * np.ones((stamps_diam, 1))))
                stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            stack = np.hstack((stack, stamps[i]))
            prev_time = MJD[i]
        return stack

    def print_stamps(self, MJD, obj, save_filename='', SN_found=False):
        """

        :param MJD:
        :param obj:
        :param save_filename:
        :param SN_found:
        :return:
        """

        num_graphs = 9

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.imshow(self.stack_stamps(obj['science'], MJD), vmin=0, vmax=600, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        plt.title(
            'Science image, position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0))
        plt.imshow(self.stack_stamps(obj['psf'], MJD), vmin=0, vmax=0.05, cmap='Greys_r', interpolation='none')
        plt.title('PSF')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (2, 0))
        plt.imshow(self.stack_stamps(obj['obs_flux'], MJD), vmin=-200, vmax=3000, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (3, 0))
        plt.imshow(self.stack_stamps(obj['obs_var_flux'], MJD), vmin=0, vmax=300, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux variance')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (4, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 0, :], MJD), vmin=-200, vmax=3000, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (5, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 1, :], MJD), vmin=-500, vmax=500, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux velocity')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (6, 0))
        plt.imshow(self.stack_stamps(-obj['pixel_flags'], MJD), vmin=-1, vmax=0, cmap='Greys_r', interpolation='none')
        plt.title('Pixel Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (7, 0))
        plt.imshow(self.stack_stamps(obj['group_flags'], MJD), vmin=-1, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Group Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (8, 0))
        plt.imshow(self.stack_stamps(obj['base_mask'], MJD), vmin=0, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Base mask')
        plt.axis('off')

        plt.tight_layout()

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_stamps', bbox_inches='tight')
        plt.close(this_fig)

    def print_space_states(self, MJD, obj, posY=-1, posX=-1, save_filename='', SN_found=False):
        """

        :param MJD:
        :param obj:
        :param posY:
        :param posX:
        :param save_filename:
        :param SN_found:
        :return:
        """

        if posY == -1:
            posY = self.obs_rad
        if posX == -1:
            posX = self.obs_rad

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.plot(obj['state'][:, 0, posY, posX], obj['state'][:, 1, posY, posX], 'b.-', label='Estimation')
        plt.plot(obj['obs_flux'][:, posY, posX], np.diff(np.concatenate((np.zeros(1), obj['obs_flux'][:, posY, posX]))),
                 'r.-', label='Observation', alpha=0.25)
        plt.grid()
        plt.plot([500, 500, 3000], [1000, 150, 0], 'k-', label='Thresholds')
        plt.legend(loc=0, fontsize='small')
        plt.plot([500, 3000], [150, 0], 'k-')
        plt.xlim(-500, 3000)
        plt.ylim(-500, 1000)
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_space_states', bbox_inches='tight')
        plt.close(this_fig)

    def print_all_space_states(self, fig, MJD, obj, sn, NUO, SN_found, save_filename=''):
        """

        :param fig:
        :param MJD:
        :param obj:
        :param sn:
        :param NUO:
        :param SN_found:
        :param save_filename:
        :return:
        """

        posY = self.obs_rad
        posX = self.obs_rad

        if NUO:
            if sn == 32:
                return
            plot_color = 'r.-'
        elif SN_found:
            plot_color = 'b.-'
        elif sn < 36:
            plot_color = 'g.-'
        else:
            return

        plt.figure(fig)
        plt.plot(obj['state'][:, 0, posY, posX], obj['state'][:, 1, posY, posX], plot_color, label='Estimation',
                 alpha=0.2)

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_space_states', bbox_inches='tight')
        # plt.close('all')

        return
