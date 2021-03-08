__author__ = "Eric Dose :: Albuquerque"

# Python core packages:
from math import sqrt, floor, ceil, log, pi

# External packages:
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

# Author's packages:
from astropak.image import FITS
from astropak.reference import FWHM_PER_SIGMA

TOP_DIRECTORY = 'C:/Astro/Images/Borea Photrix'


# R_DISC altered 10 -> 9 Aug 16 2019 for new L-500 mount.
# This is here for safety only--normally, user would pass in values derived from some .ini file
#     with values specific to a specific photometry application.
R_DISC = 9  # for aperture photometry, likely to be adaptive (per image) later.
R_INNER = 15  # "
R_OUTER = 20  # "
SUBIMAGE_MARGIN = 1.5  # subimage pixels around outer annulus, for safety


class Image:
    """
    Holds an astronomical image and apertures for photometric processing.  ALL TESTS OK 2020-10-27.
    Contains a FITS object, but doesn't know of its implementation and doesn't alter it.
    """
    def __init__(self, fits_object, aperture_radii_pixels=(R_DISC, R_INNER, R_OUTER)):
        """  Main constructor when FITS object is already available.
        :param fits_object: an object of the FITS class (this module).
        :param aperture_radii_pixels: the 3 radii defining the center disc, and the inner and outer radii
                   of the circular sky annulus around it, in pixels. [3-tuple of floats]
        """
        self.fits = fits_object
        self.top_directory = fits_object.top_directory
        self.rel_directory = fits_object.rel_directory
        self.plate_solution = fits_object.plate_solution
        self.image = self.fits.image
        self.xsize = self.image.shape[0]
        self.ysize = self.image.shape[1]
        self.aperture_radii_pixels = aperture_radii_pixels
        self.apertures = dict()  # initially empty dictionary of Aperture objects
        self.df_punches = pd.DataFrame()

    @classmethod
    def from_fullpath(cls, fullpath):
        """ Alternate constructor when FITS-file fullpath is available but not the FITS object itself.
            Will load FITS object and call regular constructor.
        """
        this_fits = FITS(fullpath)
        return Image(this_fits)

    @classmethod
    def from_fits_path(cls, top_directory=TOP_DIRECTORY, rel_directory=None, filename=None):
        """ Alternate constructor that starts by fetching the FITS object via its given filename.
            Will load FITS object and call regular constructor.
        """
        this_fits = FITS(top_directory, rel_directory, filename)
        return Image(this_fits)

    def add_aperture(self, star_id, x0, y0):
        """
        Make one aperture from position (x,y) in image, refine its sub-pixel position by
            computing its bkgd-adjusted flux centroid, and append the aperture to self.apertures
            Will replace if aperture already exists for this starID.
        :param star_id: this aperture's name, e.g., '114_1', 'ST Tri'. Unique to this Image [string]
        :param x0: initial x position of aperture center (will be refined) [float]
        :param y0: initial y position of aperture center (will be refined) [float]
        :return: [None]
        """
        if len(self.df_punches) >= 1:
            df_ap_punches = self.df_punches.loc[self.df_punches['StarID'] == star_id, :]
        else:
            df_ap_punches = None
        self.apertures[star_id] = Aperture(self, star_id, x0, y0, df_ap_punches)
        self._recenter_aperture(star_id)

    def add_punches(self, df_punches):
        """
        Add all punches to this Image's dataframe of punches, then update all affected apertures.
        :param df_punches: new punches, columns=[StarID, dNorth, dEast] [pandas DataFrame]
        :return: [None]
        """
        # Apply punches if any; then for any apertures affected:
        self.df_punches = self.df_punches.append(df_punches)  # simple append, duplicates OK.
        if len(self.df_punches) >= 1:
            ap_names_affected = set(df_punches['StarID'])
        else:
            ap_names_affected = []

        # Replace all affected apertures (incl. updating centroid and results):
        for ap_name in ap_names_affected:
            if ap_name in self.apertures:
                ap_previous = self.apertures[ap_name]
                self.add_aperture(ap_name, ap_previous.xcenter, ap_previous.ycenter)  # replace it.
            else:
                print('>>>>> Warning: df_punch StarID \'' + ap_name +
                      '\' is not a valid aperture name in ' + self.fits.filename + '.')

    def results_from_aperture(self, star_id):
        """
        Return tuple of best positiion, fluxes etc for this aperture.
        :param star_id: which aperture [string]
        :return: Series of results [indexed pandas Series of floats]
        """
        ap = self.apertures[star_id]
        return pd.Series({'r_disc': ap.r_disc,
                          'r_inner': ap.r_inner,
                          'r_outer': ap.r_outer,
                          'n_disc_pixels': ap.n_disc_pixels,
                          'n_annulus_pixels': ap.n_annulus_pixels,
                          'annulus_flux': ap.annulus_flux,
                          'annulus_flux_sigma': ap.annulus_flux_sigma,
                          'net_flux': ap.net_flux,
                          'net_flux_sigma': ap.net_flux_sigma,
                          'x_centroid': ap.x_centroid,
                          'y_centroid': ap.y_centroid,
                          'fwhm': ap.fwhm,
                          'x1024': ap.x1024,
                          'y1024': ap.y1024,
                          'vignette': ap.vignette,
                          'sky_bias': ap.sky_bias,
                          'max_adu': ap.max_adu})

    def _recenter_aperture(self, ap_name, max_cycles=2, pixels_convergence=0.05):
        """
        For one Aperture object, reset center to previously calculated centroid, update entry in
            Image's dict of Aperture objects.
        :param ap_name:
        :param max_cycles: max number of recentering cycles [int]
        :param pixels_convergence: movement by fewer pixels than this stops the refinement. [float]
        :return: [None]
        """
        for i_cycle in range(max_cycles):
            ap = self.apertures[ap_name]
            x_previous, y_previous = ap.xcenter, ap.ycenter
            x_new, y_new = ap.x_centroid, ap.y_centroid
            distance_to_move = sqrt((x_new - x_previous) ** 2 + (y_new - y_previous) ** 2)
            if distance_to_move <= pixels_convergence:
                break
            # Here, the movement distance warrants a new Aperture object:
            self.apertures[ap_name] = ap.yield_recentered()

    def _recenter_all_apertures(self):
        for ap_name in self.apertures.keys():
            self._recenter_aperture(ap_name)


class Aperture:
    """ Used directly only by class Image. Contains everything about one aperture.
       TESTS OK 2020-10-27 (tested together with class Image). """
    def __init__(self, image_obj, star_id, x0, y0, df_punches=None):
        """
        :param image_obj: Image to which this Aperture applies [Image class object]
        :param star_id: name of this aperture [string]
        :param x0: initial x center, in pixels [float]
        :param y0: initial y center, in pixels [float]
        :param df_punches: one row for each punch columns=[StarID, dNorth, dEast].
            Only rows with StarID matching star_id will be applied. [pandas DataFrame]
        """
        self.image_obj = image_obj  # reference to Image class object for this aperture.
        self.image = image_obj.image  # reference to (x,y) array holding the image data.
        self.star_id = star_id
        self.xcenter = float(x0)
        self.ycenter = float(y0)
        self.df_punches = None  # default if no punch lines available for this aperture.
        if df_punches is not None:
            if len(df_punches) >= 1:
                self.df_punches = df_punches.loc[df_punches['StarID'] == star_id, :]
        self.r_disc, self.r_inner, self.r_outer = image_obj.aperture_radii_pixels

        # Aperture evaluation fields, with default (no-flux) values:
        self.n_disc_pixels, self.n_annulus_pixels = 0, 0
        self.net_flux = 0.0
        self.net_flux_sigma = 0.0
        self.annulus_flux = 0.0
        self.annulus_flux_sigma = 0.0
        self.sn = 0.0
        self.x_centroid = self.xcenter
        self.y_centroid = self.ycenter
        self.fwhm = 0.0
        self.sky_bias = 0.0
        self.max_adu = 0.0

        # Compute needed boundaries of subimage around this aperture:
        image_xsize, image_ysize = self.image.shape
        test_radius = self.r_outer + SUBIMAGE_MARGIN
        xlow = int(floor(self.xcenter - test_radius))
        xhigh = int(ceil(self.xcenter + test_radius))
        ylow = int(floor(self.ycenter - test_radius))
        yhigh = int(ceil(self.ycenter + test_radius))

        # Compute whether needed subimage will fall entirely within image, or not:
        subimage_within_image = (xlow >= 0) & (xhigh <= image_xsize - 1) & \
                                (ylow >= 0) & (yhigh <= image_ysize - 1)

        # Compute values only if subimage entirely contained in current image:
        if subimage_within_image:
            self.subimage = self.image[xlow:xhigh + 1, ylow:yhigh + 1].copy()

            # Construct mask arrays to represent disc and annulus (both same shape as subimage):
            nx = xhigh - xlow + 1  # number of columns in subimage.
            ny = yhigh - ylow + 1  # number of rows in subimage.
            self.ygrid, self.xgrid = np.meshgrid(ylow + np.arange(ny), xlow + np.arange(nx))
            dx = self.xgrid - self.xcenter
            dy = self.ygrid - self.ycenter
            dist2 = dx**2 + dy**2
            self.disc_mask = np.clip(np.sign(self.r_disc**2 - dist2), 0.0, 1.0)
            inside_outer_edge = np.clip(np.sign(self.r_outer**2 - dist2), 0.0, 1.0)
            outside_inner_edge = np.clip(np.sign(dist2 - self.r_inner**2), 0.0, 1.0)
            self.annulus_mask = inside_outer_edge * outside_inner_edge

            # Apply punches:
            if df_punches is not None:
                if len(df_punches) >= 1:
                    self._apply_punches(image_obj.plate_solution)  # only punches for this aperture.

            # Evaluate and store several new fields:
            self.evaluate()
            del self.subimage, self.xgrid, self.ygrid, self.disc_mask, self.annulus_mask

        # Add other fields useful to calling code:
        image_center_x = self.image.shape[0] / 2.0
        image_center_y = self.image.shape[1] / 2.0
        self.x1024 = (self.xcenter - image_center_x) / 1024.0
        self.y1024 = (self.ycenter - image_center_y) / 1024.0
        self.vignette = self.x1024**2 + self.y1024**2  # no sqrt...meant to be parabolic term

    def evaluate(self):
        """
        Compute and several fields in this Aperture object. Put them in Aperture object.
        :return: [None]
        """
        self.n_disc_pixels = np.sum(self.disc_mask)
        self.n_annulus_pixels = np.sum(self.annulus_mask)
        self.annulus_flux = self._eval_sky_005()  # average adus / pixel, sky background
        estimated_background = self.n_disc_pixels * self.annulus_flux
        disc_values = np.ravel(self.subimage[self.disc_mask > 0])  # only values in mask.
        self.max_adu = np.max(disc_values)
        this_net_flux = np.sum(disc_values) - estimated_background
        if this_net_flux > 0:
            self.net_flux = this_net_flux
            gain = 1.57  # TODO: this should come from Instrument object.
            annulus_values = np.ravel(self.subimage[self.annulus_mask > 0])
            self.annulus_flux_sigma = np.std(annulus_values)
            # net_flux_sigma equation after APT paper, PASP 124, 737 (2012), but pi/2 in 3rd term
            # set to 1 as pi/2 seems hard to justify, and as 1 gives S/N closer to VPhot's values.
            self.net_flux_sigma = sqrt((self.net_flux / gain) +
                                       (self.n_disc_pixels * self.annulus_flux_sigma**2) +
                                       1.0 * ((self.n_disc_pixels*self.annulus_flux_sigma)**2 /
                                              self.n_annulus_pixels))
            self.sn = self.net_flux / self.net_flux_sigma
            # Compute centroid (x,y) of net flux:
            net_flux_grid = self.disc_mask * (self.subimage - self.annulus_flux)
            normalizor = np.sum(net_flux_grid)
            if (self.x_centroid is not None) and (self.y_centroid is not None):
                self.xcenter = self.x_centroid  # new subimage center
                self.ycenter = self.y_centroid  # "
            self.x_centroid = np.sum(net_flux_grid * self.xgrid) / normalizor
            self.y_centroid = np.sum(net_flux_grid * self.ygrid) / normalizor

            # Other evaluation results:
            self.fwhm = self._eval_fwhm()
            sky_flux_bias = self.n_disc_pixels * self.annulus_flux_sigma
            self.sky_bias = abs(-2.5 * (sky_flux_bias / self.net_flux) / log(10.0))

    def yield_recentered(self):
        x_new, y_new = self.x_centroid, self.y_centroid
        return Aperture(self.image_obj, self.star_id, x_new, y_new, self.df_punches)

    def _apply_punches(self, plate_solution):
        """
        Apply punches to (remove appropriate pixels from) this Aperture's annulus mask.
        :param plate_solution:
        :return: [None]
        """
        dnorth_dx = 3600.0 * plate_solution['CD2_1']  # in arcseconds northward /pixel
        dnorth_dy = 3600.0 * plate_solution['CD2_2']  # "
        deast_dx = 3600.0 * plate_solution['CD1_1']   # in arcseconds eastward (not RA) /pixel
        deast_dy = 3600.0 * plate_solution['CD1_2']   # "
        ann_mask_new = self.annulus_mask.copy()  # to begin.
        for dnorth, deast in zip(self.df_punches['dNorth'], self.df_punches['dEast']):
            coefficients = np.array([[dnorth_dx, dnorth_dy], [deast_dx, deast_dy]])
            dep_vars = np.array([dnorth, deast])
            solution = np.linalg.solve(coefficients, dep_vars)
            dx_punch, dy_punch = solution[0], solution[1]
            x_punch = self.xcenter + dx_punch
            y_punch = self.ycenter + dy_punch
            x_dist = self.xgrid - x_punch  # x distance from center
            y_dist = self.ygrid - y_punch  # y distance from center
            dist2 = x_dist**2 + y_dist**2
            punch_mask = np.clip(np.sign(dist2 - self.r_disc**2), 0.0, 1.0)
            ann_mask_new = ann_mask_new * punch_mask  # do the punch (pixels within punch set to 0).
        self.annulus_mask = ann_mask_new

    def _eval_sky_005(self):
        """
        Winning sky-background measurement strategy of 2015 tournament of strategies.
        Insensitive to cosmic rays and background stars in or near the annulus.
        :return robust estimate of sky background in adu/pixel [float]
        """
        slice_list = self._make_sky_slices(n_slices=12, method='trimmed_mean')
        sky_adu = trim_mean(slice_list, proportiontocut=0.3)
        return sky_adu

    def _make_sky_slices(self, n_slices=12, method='trimmed_mean'):
        radians_per_slice = (2.0 * pi) / n_slices
        min_pixels_per_slice = 0.5 * (self.n_annulus_pixels / n_slices)
        angle_grid = np.arctan2(self.ygrid-self.ycenter, self.xgrid-self.xcenter)  # -pi to +pi
        slice_list = []
        for i_slice in range(n_slices):
            # Radians delimiting this slice:
            angle_min = i_slice * radians_per_slice - pi
            angle_max = (i_slice + 1) * radians_per_slice - pi
            above_min = np.clip(np.sign(angle_grid - angle_min), 0.0, 1.0)
            below_max = np.clip(np.sign(angle_max - angle_grid), 0.0, 1.0)
            slice_mask = above_min * below_max * self.annulus_mask
            n_slice_pixels = np.sum(slice_mask)
            if n_slice_pixels >= min_pixels_per_slice:
                slice_values = np.ravel(self.subimage[slice_mask > 0])  # only values in mask.
                slice_mean = trim_mean(slice_values, 0.4)
                slice_list.append(slice_mean)
        return slice_list

    def _eval_fwhm(self):
        # TODO: Probably need a better FWHM algorithm.
        """
        Returns estimate of Full Width at Half-Maximum from mean dist2 (=2*sigma^2) of net flux.
        This algorithm may be replaced later:
            overestimates FWHM compared to MaxIm, PinPoint, and sometimes to even visual inspection.
        :return: estimate of FWHM in pixels.
        """
        dx = self.xgrid - self.x_centroid
        dy = self.ygrid - self.y_centroid
        dist2 = dx ** 2 + dy ** 2
        net_flux_xy = (self.disc_mask * (self.subimage - self.annulus_flux))
        mean_dist2 = max(0.0, np.sum(net_flux_xy * dist2) / np.sum(net_flux_xy))
        sigma = sqrt(mean_dist2 / 2.0)
        # this math is verified 20170723, but yields larger FWHM than does MaxIm.
        return FWHM_PER_SIGMA * sigma