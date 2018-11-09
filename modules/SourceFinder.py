from modules.utils import *

from modules.DataContent import DataContent

class SourceFinder(object):

    def __init__(self, flux_thresh, flux_rate_thresh, rate_satu, accum_neg_flux_depth=4, prev_fluxes = None,
                 accum_med_flux_depth=3,image_size=(4094, 2046), n_consecutive_obs=4):
        """

        :param flux_thresh:
        :param flux_rate_thresh:
        :param rate_satu:
        :param n_consecutive_obs:
        :param image_size:
        """
        self.image_size = image_size
        self.flux_thresh = flux_thresh
        self.flux_rate_thresh = flux_rate_thresh
        self.rate_satu = rate_satu
        self.accum_neg_flux = np.zeros(tuple([accum_neg_flux_depth]) + image_size, dtype=bool)
        self.accum_neg_flux_depth = accum_neg_flux_depth
        self.n_obs_prev = 0
        self.n_consecutive_obs = n_consecutive_obs
        self.any_pixels = False


        if prev_fluxes != None:
            self.n_obs_prev = len(prev_fluxes)#n_obs < 4
            self.accum_neg_flux[0:self.n_obs_prev, :] = prev_fluxes
        #self.n_consecutive_obs = n_consecutive_obs
        self.accum_compliant_pixels = np.zeros(tuple([n_consecutive_obs]) + image_size, dtype=bool)
        self.accum_median_flux = np.zeros(tuple([accum_med_flux_depth]) + image_size)
        self.median_rejection = np.zeros(image_size, dtype=bool)
        self.accum_med_flux_depth=accum_med_flux_depth
        #self.median_rejection = np.zeros(image_size, dtype=bool)


    def pixel_discard(self, science, state, state_cov, dil_mask, median_reject):
        """

        :param science: Science image
        :param state: State (image size array) previously obtained by Kalman filter
        :param state_cov: Covariance array determined by kalman filter
        :param dil_mask: dilation mask
        :param median_rejection: stimated median rejection
        :return:
        """
        n_conditions = 7



        science_median = subsampled_median(science, self.image_size, 20)
        pixel_conditions = np.zeros(tuple([n_conditions]) + self.image_size, dtype=bool) #Every pixel
        pixel_flags = np.zeros(self.image_size, dtype=int)
        pixel_conditions[:] = False
        pixel_flags[:] = 0

        pixel_conditions[0, :] = state[0, :] > self.flux_thresh
        pixel_conditions[1, :] = state[1, :] > (self.flux_rate_thresh * (
                self.rate_satu - np.minimum(state[0, :], self.rate_satu)) / self.rate_satu)
        pixel_conditions[2, :] = science > science_median + 5
        pixel_conditions[3, :] = state_cov[0, :] < 150.0  # check value
        pixel_conditions[4, :] = state_cov[2, :] < 150.0
        pixel_conditions[5, :] = np.logical_not(dil_mask)
        pixel_conditions[6, :] = np.logical_not(median_reject)

        #If pixels don't satisfy these conditions are labeled
        for i in range(n_conditions):
            pixel_flags[np.logical_not(pixel_conditions[i, :])] += 2 ** i

        return pixel_flags


    def grouping_pixels(self, pixel_flags, o):
        
        self.accum_compliant_pixels[o % self.n_consecutive_obs, :] = pixel_flags == 0
        #self.accum_compliant_pixels[o % self.n_consecutive_alerts, :] = self.pixel_flags == 0

        alert_pixels = np.all(self.accum_compliant_pixels, 0)
        self.data_content = DataContent()

        if not np.any(alert_pixels):
            return

        else:

            labeled_image, nr_objects = mh.label(pixel_flags==0, Bc=np.ones((3, 3), dtype=int))
            labeled_image_coords = np.nonzero(labeled_image)
            labeled_image_values = labeled_image[np.nonzero(labeled_image)]
            labeled_image_coords = np.array(labeled_image_coords).T

            sorted_args = np.argsort(labeled_image_values)
            labeled_image_values = labeled_image_values[sorted_args]
            labeled_image_coords = labeled_image_coords[sorted_args, :]

            n_neighboring_pixels = labeled_image_values[-1]

            self.data_content.set_mid_coords(n_neighboring_pixels)

            for i in range(n_neighboring_pixels):

                self.data_content.pixel_coords += [labeled_image_coords[labeled_image_values == i + 1, :]]
                self.data_content.pixel_mid_coords[i, :] = np.round(np.mean(self.data_content.pixel_coords[i], 0))

    def filter_groups(self, science, flux, var_flux, state, base_mask, median_reject=None):

            n_pixel_groups = self.data_content.group_info(self.image_size)

            for i in range(n_pixel_groups):

                posY, posX = self.data_content.pixel_mid_coords[i, :]#self.PGData['mid_coords'][i, :]

                # Discard groups with negative flux around (bad substractions)
                NNFR = 4
                a, b = posY - NNFR, posY + NNFR + 1
                c, d = posX - NNFR, posX + NNFR + 1
                if np.any(self.accum_neg_flux[:, a:b, c:d]):
                    self.data_content.group_flags[i] += 1

                # Local Maximum Radius in Science Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                scienceLM = mh.regmax(science[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(scienceLM[1:-1, 1:-1]):
                    self.data_content.group_flags[i] += 2#self.PGData['group_flags'][i] += 2

                # Local Maxima Radius in Flux Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                fluxLM = mh.regmax(flux[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(fluxLM[1:-1, 1:-1]):
                    self.data_content.group_flags[i] += 4#self.PGData['group_flags'][i] += 4

                # Local Maxima Radius in Estimated Flux Velocity Image
                LMSR = 3
                a, b = posY - LMSR + 1, posY + LMSR + 2
                c, d = posX - LMSR + 1, posX + LMSR + 2
                velLM = mh.regmax(state[1, a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
                if not np.any(velLM[1:-1, 1:-1]):
                    self.data_content.group_flags[i] += 8#self.PGData['group_flags'][i] += 8

                # Above local science median
                ASMR = 3
                a, b = posY - ASMR, posY + ASMR + 1
                c, d = posX - ASMR, posX + ASMR + 1
                if not (science[posY, posX] > np.median(science[a:b, c:d]) + 15):
                    self.data_content.group_flags[i] += 16#self.PGData['group_flags'][i] += 16

                # Brightest Pixel on stamps (flux and science)
                BPOS = 10
                a, b = posY - BPOS, posY + BPOS + 1
                c, d = posX - BPOS, posX + BPOS + 1
                brightPixels = np.logical_or(flux[a:b, c:d] > 2 * flux[posY, posX],
                                             science[a:b, c:d] > 2 * science[posY, posX])
                if np.any(brightPixels):
                    self.data_content.group_flags[i] += 32#self.PGData['group_flags'][i] += 32

                # Center over mask
                if base_mask[posY, posX] > 0:
                    self.data_content.group_flags[i] += 64

                # Center over median-rejected pixel
                if median_reject[posY, posX]:
                    self.data_content.group_flags[i] += 128

                # flux variance
                if var_flux[posY, posX] > 250.0:
                    self.data_content.group_flags[i] += 256

                self.data_content.group_flags_map[self.data_content.pixel_coords[i][:, 0],
                                                  self.data_content.pixel_coords[i][:, 1]] = \
                    self.data_content.group_flags[i]


    def draw_complying_pixel_groups(self, science, state, state_cov, base_mask,
                                    dil_mask, flux, var_flux,
                                     mjd, field, ccd, path_, o, median_reject=None, last=False):


        self.accum_neg_flux[o % self.accum_neg_flux_depth, :] = flux < 0
        self.median_rejection, self.accum_median_flux = median_rejection_calc(self.median_rejection,
                                                                                  self.accum_median_flux,
                                                                                  self.accum_med_flux_depth, flux, o)

        pixel_flags = self.pixel_discard(science, state, state_cov, dil_mask, median_reject)
        self.grouping_pixels(pixel_flags, o)
        #if self.any_pixels:
        self.filter_groups(science, flux, var_flux, state, base_mask, median_reject=median_reject)



        if not last:
            self.data_content.save_results(path_, field, ccd, mjd)
        else:
            self.data_content.save_results(path_, field, ccd, mjd, state=state,
                                           state_cov=state_cov, save_state_info=True)
