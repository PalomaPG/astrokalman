from modules.utils import *

from modules.DataContent import DataContent

class SourceFinder(object):

    def __init__(self, flux_thresh, flux_rate_thresh, rate_satu, accum_neg_flux_depth=4,
                 accum_med_flux_depth=3,image_size=(4094, 2046), n_consecutive_obs=4):
        """

        :param flux_thresh:
        :param flux_rate_thresh:
        :param rate_satu:
        :param n_consecutive_obs:
        :param image_size:
        """
        self.image_size = image_size
        self.n_consecutive_obs = n_consecutive_obs
        self.n_conditions = 7
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + image_size, dtype=bool)
        self.pixel_flags = np.zeros(image_size, dtype=int)
        print(self.pixel_flags)
        #self.n_consecutive_obs = n_consecutive_obs
        self.accum_compliant_pixels = np.zeros(tuple([n_consecutive_obs]) + image_size, dtype=bool)


        self.flux_thresh = flux_thresh
        self.flux_rate_thresh = flux_rate_thresh
        self.rate_satu = rate_satu
        #self.median_rejection = np.zeros(image_size, dtype=bool)
        self.CandData = []

        self.median_rejection = np.zeros(image_size, dtype=bool)
        self.accum_neg_flux = np.zeros(tuple([accum_neg_flux_depth]) + image_size, dtype=bool)
        self.accum_neg_flux_depth = accum_neg_flux_depth
        self.accum_med_flux_depth = accum_med_flux_depth
        self.accum_median_flux = np.zeros(tuple([accum_med_flux_depth]) + image_size)



    def pixel_discard(self, science, state, state_cov, dil_mask, o):
        """

        :param science: Science image
        :param state: State (image size array) previously obtained by Kalman filter
        :param state_cov: Covariance array determined by kalman filter
        :param dil_mask: dilation mask
        :param median_rejection: stimated median rejection
        :return:
        """

        science_median = subsampled_median(science, self.image_size, 20)
        self.pixel_conditions[:] = False
        type(self.pixel_flags)



        self.pixel_flags[:] = 0

        self.pixel_conditions[0, :] = state[0, :] > self.flux_thresh
        self.pixel_conditions[1, :] = state[1, :] > (self.flux_rate_thresh * (
                self.rate_satu - np.minimum(state[0, :], self.rate_satu)) / self.rate_satu)
        self.pixel_conditions[2, :] = science > science_median + 5
        self.pixel_conditions[3, :] = state_cov[0, :] < 150.0  # check value
        self.pixel_conditions[4, :] = state_cov[2, :] < 150.0
        self.pixel_conditions[5, :] = np.logical_not(dil_mask)
        self.pixel_conditions[6, :] = np.logical_not(self.median_rejection)

        #If pixels don't satisfy these conditions are labeled
        for i in range(self.n_conditions):
            self.pixel_flags[np.logical_not(self.pixel_conditions[i, :])] += 2 ** i

        self.accum_compliant_pixels[o % self.n_consecutive_obs, :] = (self.pixel_flags == 0)



    def grouping_pixels(self):

        self.PGData = {}  # Pixel group data
        self.PGData['pixel_coords'] = []

        alert_pixels = np.all(self.accum_compliant_pixels, 0)
        #self.data_content = DataContent()

        if not np.any(alert_pixels):
            self.PGData['mid_coords'] = np.zeros((0, 2), dtype=int)
            return

        else:

            labeled_image, nr_objects = mh.label(alert_pixels, Bc=np.ones((3, 3), dtype=int))
            labeled_image_coords = np.nonzero(labeled_image)
            labeled_image_values = labeled_image[np.nonzero(labeled_image)]
            labeled_image_coords = np.array(labeled_image_coords).T

            sorted_args = np.argsort(labeled_image_values)
            labeled_image_values = labeled_image_values[sorted_args]
            labeled_image_coords = labeled_image_coords[sorted_args, :]

            n_neighboring_pixels = labeled_image_values[-1]

            self.PGData['mid_coords'] = np.zeros((n_neighboring_pixels, 2), dtype=int)
            #data_content.set_mid_coords(n_neighboring_pixels)

            for i in range(n_neighboring_pixels):
                self.PGData['pixel_coords'] += [labeled_image_coords[labeled_image_values == i + 1, :]]
                self.PGData['mid_coords'][i, :] = np.round(np.mean(self.PGData['pixel_coords'][i], 0))

    def filter_groups(self, science, flux, var_flux, state, base_mask):

            n_pixel_groups = self.PGData['mid_coords'].shape[0]
            #data_content.group_info(self.image_size)

            self.PGData['group_flags'] = np.zeros(n_pixel_groups, dtype=int)
            self.PGData['group_flags_map'] = -np.ones((4094, 2046), dtype=int)

            for i in range(n_pixel_groups):

                posY, posX =self.PGData['mid_coords'][i, :] # data_content.pixel_mid_coords[i, :]

                # Discard groups with negative flux around (bad substractions)
                NNFR = 4
                a, b = posY - NNFR, posY + NNFR + 1
                c, d = posX - NNFR, posX + NNFR + 1
                if np.any(self.accum_neg_flux[:, a:b, c:d]):
                    self.PGData['group_flags'][i] += 1

                # Local Maximum Radius in Science Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                scienceLM = mh.regmax(science[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(scienceLM[1:-1, 1:-1]):
                    self.PGData['group_flags'][i]+= 2 #self.PGData['group_flags'][i] += 2

                # Local Maxima Radius in Flux Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                fluxLM = mh.regmax(flux[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(fluxLM[1:-1, 1:-1]):
                    self.PGData['group_flags'][i]+= 4 #self.PGData['group_flags'][i] += 4

                # Local Maxima Radius in Estimated Flux Velocity Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                velLM = mh.regmax(state[1, a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(velLM[1:-1, 1:-1]):
                    self.PGData['group_flags'][i] += 8#self.PGData['group_flags'][i] += 8

                # Above local science median
                ASMR = 3
                a, b = posY - ASMR, posY + ASMR + 1
                c, d = posX - ASMR, posX + ASMR + 1
                if not (science[posY, posX] > np.median(science[a:b, c:d]) + 15):
                    self.PGData['group_flags'][i] += 16 #self.PGData['group_flags'][i] += 16

                # Brightest Pixel on stamps (flux and science)
                BPOS = 10
                a, b = posY - BPOS, posY + BPOS + 1
                c, d = posX - BPOS, posX + BPOS + 1
                brightPixels = np.logical_or(flux[a:b, c:d] > 2 * flux[posY, posX],
                                             science[a:b, c:d] > 2 * science[posY, posX])
                if np.any(brightPixels):
                    self.PGData['group_flags'][i] += 32 #self.PGData['group_flags'][i] += 32

                # Center over mask
                if base_mask[posY, posX] > 0:
                    self.PGData['group_flags'][i] += 64

                # Center over median-rejected pixel
                if self.median_rejection[posY, posX]:
                    self.PGData['group_flags'][i] += 128

                # flux variance
                if var_flux[posY, posX] > 250.0:
                    self.PGData['group_flags'][i] += 256

                    self.PGData['group_flags_map'][self.PGData['pixel_coords'][i][:, 0],
                                                   self.PGData['pixel_coords'][i][:, 1]] = self.PGData['group_flags'][i]

    def update_candidates(self, o):
        """

        :param o:
        :return:
        """
        cand_mid_coords = self.PGData['mid_coords'][self.PGData['group_flags'] == 0, :]

        for i in range(cand_mid_coords.shape[0]):

            for c in range(len(self.CandData) + 1):
                if c == len(self.CandData):
                    # New candidate
                    new_cand = {}
                    new_cand['coords'] = cand_mid_coords[i, :]
                    new_cand['epochs'] = [o]
                    self.CandData += [new_cand]
                else:
                    # Part of a previous candidate?
                    if (np.sqrt(np.sum((cand_mid_coords[i, :] - self.CandData[c]['coords']) ** 2)) < 4.0):
                        n_epochs = len(self.CandData[c]['epochs'])
                        self.CandData[c]['coords'] = (self.CandData[c]['coords'] * n_epochs + cand_mid_coords[i, :]) / (
                                    n_epochs + 1)
                        self.CandData[c]['epochs'] += [o]
                        break
        print(self.CandData)

    def check_candidates(self, SN_index, SN_pos):
        """

        :param RD: RunData instance
        :return:
        """
        # print('# of candidates found')
        # print(len(self.CandData))
        self.NUO = 0  # Number of Unknown Objects

        if SN_index >= 0:
            self.SN_found = False

            for i in range(len(self.CandData)):
                distance = np.sqrt(np.sum((self.CandData[i]['coords'] - SN_pos) ** 2))

                if distance < 4.0:
                    self.CandData[i]['status'] = 0
                    print('FOUND')
                    self.SN_found = True
                else:
                    self.CandData[i]['status'] = -1
                    self.NUO += 1
        else:
            self.NUO = len(self.CandData)
            for i in range(len(self.CandData)):
                self.CandData[i]['status'] = -1



    def draw_complying_pixel_groups(self, science, state, state_cov, base_mask,
                                    dil_mask, flux, var_flux, o, SN_index, SN_pos):

        self.accum_neg_flux[o % self.accum_neg_flux_depth, :] = flux < 0
        self.median_rejection, self.accum_median_flux = median_rejection_calc(self.median_rejection,
                                                                                  self.accum_median_flux,
                                                                                  self.accum_med_flux_depth, flux, o)

        self.pixel_discard(science, state, state_cov, dil_mask, o)
        self.grouping_pixels()
        #if self.any_pixels:
        self.filter_groups(science, flux, var_flux, state, base_mask)
        self.update_candidates(o)
        self.check_candidates(SN_index, SN_pos)


