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



_____Old_AP_CLASS_and_related_classes________________________________________ = 0


class Old_Ap:
    """ General-purpose square slice of an image, mostly meant for aperture photometry.
        This is also the parent class and engine for more specifically defined aperture classes, especially
            ApStationary (esp. for stars) and ApMoving (esp. for Minor Planets / asteroids).
        This is a much more versatile successor to mp_phot.util's class Square.
        All Old_Ap objects are IMMUTABLE once constructed; recentering generates and returns new object.

        Masks are required for this class to define which pixels are used, or the caller must specifically
            specify one or both masks are not to be used.
        Masks are foreground (light source) and background (surrounding sky). Any pixels falling outside
            the parent image range are always set to True (masked out, invalidated).
        If both masks are given as None: foreground mask is all cutout pixels, background mask is null.
        If foreground mask is given but background mask is None: foreground mask is used, and
            background mask is made the logical inverse of the foreground mask.
        If both masks are given, they are both used as given.
        Masks are boolean arrays, must match data in shape, and follow numpy's mask convention:
            mask pixel=True means corresponding data pixel is invalid ("masked away") and not used.
            mask pixel=False means data pixel is valid and used.
    """
    def __init__(self, parent, xy_center, cutout_radius, foreground_mask=None, background_mask=None):
        """ General constructor, from explicitly passed-in parent data array and 2 mask arrays.
        :param parent: the parent image array [numpy ndarray; to pass in CCDData or numpy masked array,
                   please see separate, specific constructors, below].
        :param xy_center: center pixel position in parent. This should be the best prior estimate
                   of the light source's centroid at mid-exposure, as (x,y) (not as numpy [y, x] array).
                   [2-tuple, 2-list, or 2-array of floats]
        :param cutout_radius: half-length of cutout's edge length less one, in pixels.
                   Actual size of cutout will be from cutout_radius rounded up,
                   that is, from 2 * ceil(cutout_radius) + 1.
                   So Old_Ap object size and shape are always an odd number of pixels, no exceptions. [float]
        :param foreground_mask: mask array for pixels to be counted in flux, centroid, etc.
                   Shape must exactly match shape of either the parent array or the cutout array.
                   Array True -> MASKED pixel, False -> pixel is valid and used. (numpy convention).
                   Specifying None means 'use all pixels'. [numpy ndarray of booleans, or None]
        :param background_mask: mask array for pixels to be counted in background flux, centroid, etc.
                   Shape must exactly match shape of either the parent array or the cutout array.
                   Array True -> MASKED pixel, False -> pixel is valid and used. (numpy convention).
                   Specifying None means background is assumed to be zero and has no effect.
                   [numpy ndarray of booleans, or None]
        """
        self.parent = parent.copy()
        self.xy_center = tuple(xy_center)
        self.cutout_radius = int(ceil(cutout_radius))
        self.input_foreground_mask = foreground_mask
        self.input_background_mask = background_mask
        self.messages = []

        # Construct small sub-array from parent pixels:
        self.x_offset, self.y_offset = calc_cutout_offsets(self.xy_center, cutout_radius)
        self.x_raw_low = self.x_offset
        self.x_center = self.x_offset + self.cutout_radius
        self.x_raw_high = self.x_offset + 2 * self.cutout_radius
        self.y_raw_low = self.y_offset
        self.y_center = self.y_offset + self.cutout_radius
        self.y_raw_high = self.y_offset + 2 * self.cutout_radius
        edge_pixel_count = self.cutout_radius * 2 + 1
        self.shape = (edge_pixel_count, edge_pixel_count)
        self.size = edge_pixel_count ** 2
        x_data_low = max(0, self.x_raw_low)
        x_data_high = min(parent.shape[1] - 1, self.x_raw_high)
        y_data_low = max(0, self.y_raw_low)
        y_data_high = min(parent.shape[0] - 1, self.y_raw_high)
        if (x_data_low > x_data_high) or (y_data_low > y_data_high):
            self.messages.append('Data boundaries are outside parent array. '
                                 'Old_Ap object cannot be constructed.')
            self.is_valid = False
            return

        # Make Old_Ap's data array: ensure any pixels outside parent have value np.nan:
        self.data = np.full(self.shape, fill_value=np.nan, dtype=np.double)  # template only.
        parent_data_available = self.parent[y_data_low: y_data_high + 1,
                                x_data_low: x_data_high + 1].copy()
        self.data[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
        x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = parent_data_available

        # Ensure masks are in shape of Old_Ap's data array (extract if parent-sized masks passed in),
        #    also ensure that any pixels outside parent have value True (to invalidate):
        within_parent_mask = np.full(self.shape, fill_value=True, dtype=np.bool)
        within_parent_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
        x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = False
        self.is_all_within_parent = np.all(within_parent_mask == False)

        if self.input_foreground_mask is None:
            self.foreground_mask = within_parent_mask
        else:
            self.foreground_mask = np.full(self.shape, fill_value=True, dtype=np.bool)  # template only.
            if self.input_foreground_mask.shape == self.parent.shape:
                parent_foreground_mask_available = \
                    self.input_foreground_mask[y_data_low: y_data_high + 1,
                    x_data_low: x_data_high + 1].copy()
                self.foreground_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    parent_foreground_mask_available
            elif self.input_foreground_mask.shape == self.shape:
                self.foreground_mask = np.logical_or(self.input_foreground_mask, within_parent_mask)
            else:
                self.messages.append('Foreground mask shape does not match data shape. '
                                     'Old_Ap object cannot be constructed.')
                self.is_valid = False
                return

        if self.input_background_mask is None:
            foreground_raw_inverse = np.logical_not(self.foreground_mask)
            self.background_mask = np.logical_or(foreground_raw_inverse, within_parent_mask)
        else:
            self.background_mask = np.full(self.shape, fill_value=True, dtype=np.bool)  # template only.
            if self.input_background_mask.shape == self.parent.shape:
                parent_background_mask_available = \
                    self.input_background_mask[y_data_low: y_data_high + 1,
                    x_data_low: x_data_high + 1].copy()
                self.background_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    parent_background_mask_available
            elif self.input_background_mask.shape == self.shape:
                self.background_mask = np.logical_or(self.input_background_mask, within_parent_mask)
            else:
                self.messages.append('Background mask shape does not match data shape. '
                                     'Old_Ap object cannot be constructed.')
                self.is_valid = False
                return

        self.pixel_count = self.size  # synonym.
        self.foreground_pixel_count = np.sum(self.foreground_mask == False)  # valid pixels
        self.background_pixel_count = np.sum(self.background_mask == False)  # "
        self.mask_overlap_pixel_count = np.sum(np.logical_and((self.foreground_mask == False),
                                                              (self.background_mask == False)))
        self.xy_centroid = self._calc_centroid()
        self.is_valid = True
        self.is_pristine = (len(self.messages) == 0) and self.is_valid

    def __str__(self):
        return 'Old_Ap object of x,y shape (' + str(self.shape[1]) + ', ' + str(self.shape[0]) + ')' + \
               ' from parent image of x,y shape (' + str(self.parent.shape[1]) + ', ' + \
               str(self.parent.shape[0]) + '), masks passed in directly.'

    def _calc_centroid(self):
        """ Calculate (x,y) centroid of background adjusted flux, in pixels of parent image.
         :return: centroid position of background-adjusted flux, in parent-image pixels (x,y).
                  [2-tuple of floats]
        Yes, the conventions and polarities are non-intuitive, given numpy arrays' [y,x] indexing
            as well as numpy.meshgrid()'s utterly AMBIGUOUS documentation of the identities of x and y,
            as well as PyCharm SciView's incorrectly TRANSPOSED display of numpy arrays. Arrrgggghhh.
        """
        # Calculate x,y position of centroid:
        x_grid, y_grid = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))
        background_level, _ = calc_background_value(self.data, self.background_mask)
        data_minus_background = self.data - background_level
        x_net_product = x_grid * data_minus_background
        y_net_product = y_grid * data_minus_background
        x_product_sum = np.sum(x_net_product, where=np.logical_not(self.foreground_mask))
        y_product_sum = np.sum(y_net_product, where=np.logical_not(self.foreground_mask))
        data_sum = np.sum(data_minus_background, where=np.logical_not(self.foreground_mask))
        x_centroid = self.x_offset + (x_product_sum / data_sum)
        y_centroid = self.y_offset + (y_product_sum / data_sum)
        # Calculate sigma & FWHM of flux:
        # TODO: Add facility for shape sigma/FWHM (+++ can start with the very same net_product arrays).
        # Will require a specially made 2-D Gaussian parent image (with known sigma), prob from
        #     external library, e.g., photutils.datasets.make_gaussian_sources_image() etc.
        dx = x_grid - x_centroid
        dy = y_grid - y_centroid
        dx2 = dx * dx
        dy2 = dy * dy
        dx2_product = dx2 * data_minus_background
        dy2_product = dy2 * data_minus_background
        # TODO: This 'sigma' may need to be multiplied by 1/2.
        sigma = (dx2_product + dy2_product) / data_sum
        return x_centroid, y_centroid

    def net_flux(self, gain=1):
        """ Return net ADU flux and other statistics, usually the end goal of using this class.
            Subclasses should inherit this unchanged.
        :param gain: CCD-like gain in e-/ADU. Property of camera. Needed only for
                     accurate uncertainty estimation. [float]
        :return: background-adjusted foreground (source) 'net' flux,
                 std dev uncertainty of net flux,
                 background level (sigma-clipped median), and
                 std dev uncertainty in background level (per pixel, not of background area's average).
                 [4-tuple of floats]
        """
        raw_flux = np.sum(self.data, where=np.logical_not(self.foreground_mask))
        flux_variance_from_poisson = raw_flux / gain  # from var(e-) = flux in e-.
        if self.background_pixel_count >= 2:
            background_level, background_stddev = calc_background_value(self.data, self.background_mask)
        else:
            background_level, background_stddev = 0.0, 0.0  # no effect on flux or stddev.
        background_adjusted_flux = raw_flux - self.foreground_pixel_count * background_level

        flux_variance_from_background = self.foreground_pixel_count * \
                                        ((background_stddev ** 2) / self.background_pixel_count)
        flux_variance = flux_variance_from_poisson + flux_variance_from_background
        flux_stddev = sqrt(flux_variance)
        return background_adjusted_flux, flux_stddev, background_level, background_stddev

    def make_new_object(self, new_xy_center):
        """ Make new object using new xy_center.
            For this (parent) class with masks explicitly supplied by user, and masks do not change,
                so this function is probably not very useful for the parent class Old_Ap.
            Note: this requires a cutout_radius, but subclasses should compute their own.
            For the most subclasses (e.g., PointSourceAp), masks will be recreated by the constructor,
                using new xy_center.
         """
        return Old_Ap(self.parent, new_xy_center, self.cutout_radius,
                      self.foreground_mask, self.background_mask)

    def recenter(self, max_adjustment=None, max_iterations=3):
        """
            Subclasses should inherit this unchanged.
                (Though .make_new_object() must be written specially for each subclass).
        :param max_adjustment:
        :param max_iterations:
        :return: newly recentered object. [PointSourceAp object]
        """
        previous_ap = self
        next_ap = self  # keep IDE happy.
        for i in range(max_iterations):
            previous_centroid = previous_ap.xy_centroid
            new_xy_center = previous_centroid
            next_ap = self.make_new_object(new_xy_center)
            new_centroid = next_ap.xy_centroid
            adjustment = sqrt((new_centroid[0] - previous_centroid[0])**2 +
                              (new_centroid[1] - previous_centroid[1])**2)
            if adjustment < max_adjustment:
                return next_ap
            previous_ap = next_ap
        return next_ap


class Old_PointSourceAp(Old_Ap):
    """ Standard photometric aperture for stationary point source of light, esp. for a star.
        Always makes a circular foreground mask and an annular background mask, both centered on the
            given image coordinates of the point source.
        (If we will need a background mask that bleeds to the cutout's edges rather than being restricted
            to an annulus--for example, to work close to the parent image's edges, then that will
            definitely require a new sibling class to this one so that recentering retains the mask shapes.)
    """
    # noinspection PyTypeChecker
    def __init__(self, parent, xy_center, foreground_radius, gap, background_width):
        """ Main and probably sole constructor.
        :param parent:
        :param xy_center:
        :param foreground_radius: radial size of foreground around point source, in pixels. [float]
        :param gap: width of gap, difference between radius of foreground and inside radius of
                        background annulus, in pixels. [float]
        :param background_width: width of annulus, difference between inside and outside radii
                   of background annulus, in pixels.
        """
        # Parms specific to PointSourceAp:
        self.foreground_radius = foreground_radius
        self.gap = gap
        self.background_width = background_width
        cutout_radius = int(ceil(self.foreground_radius + self.gap + self.background_width)) + 1
        cutout_size = 2 * cutout_radius + 1
        self.x_offset, self.y_offset = calc_cutout_offsets(xy_center, cutout_radius)
        xy_cutout_center = xy_center[0] - self.x_offset, xy_center[1] - self.y_offset

        foreground_mask = make_circular_mask(mask_size=cutout_size, xy=xy_cutout_center,
                                             radius=self.foreground_radius)
        radius_inside = self.foreground_radius + self.gap
        radius_outside = radius_inside + self.background_width
        background_mask_center_disc = np.logical_not(make_circular_mask(cutout_size, xy_cutout_center,
                                                                        radius_inside))
        background_mask_outer_disc = make_circular_mask(cutout_size, xy_cutout_center, radius_outside)
        background_mask = np.logical_or(background_mask_center_disc, background_mask_outer_disc)
        super().__init__(parent, xy_center, cutout_radius, foreground_mask, background_mask)

    def make_new_object(self, new_xy_center):
        """ Overrides parent-class method.
            Make new object using new xy_center.
            Masks will be recreated by the constructor, using new xy_center.
         """
        return Old_PointSourceAp(self.parent, new_xy_center,
                                 self.foreground_radius, self.gap, self.background_width)


class Old_MovingSourceAp(Old_Ap):
    """ Elongated 'pill-shaped' photometric aperture for moving point source of light,
            esp. for a minor planet/asteroid.
        (If we will need a background mask that bleeds to the cutout's edges rather than being restricted
            to a (pill-shaped) annulus--for example, to work close to the parent image's edges, that will
            definitely require a new sibling class to this one so that recentering retains the mask shapes.)
        """
    # noinspection PyTypeChecker
    def __init__(self, parent, xy_start, xy_end, foreground_radius, gap, background_width):
        """ Main and probably sole constructor.
        :param parent:
        :param xy_start:
        :param xy_end:
        :param foreground_radius:
        :param gap:
        :param background_width:
        """
        # Parms specific to MovingSourceAp:
        self.xy_start = xy_start
        self.xy_end = xy_end
        self.foreground_radius = foreground_radius
        self.gap = gap
        self.background_width = background_width
        self.xy_motion = xy_end[0] - xy_start[0], xy_end[1] - xy_start[1]

        xy_center = (xy_start[0] + xy_end[0]) / 2.0, (xy_start[1] + xy_end[1]) / 2.0
        foreground_mask_x_span = abs(xy_start[0] - xy_end[0]) + 2 * foreground_radius
        foreground_mask_y_span = abs(xy_start[1] - xy_end[1]) + 2 * foreground_radius
        max_foreground_mask_span = max(foreground_mask_x_span, foreground_mask_y_span)
        cutout_radius = int(ceil(max_foreground_mask_span / 2 + gap + background_width)) + 1
        cutout_size = 2 * cutout_radius + 1
        cutout_x_offset, cutout_y_offset = calc_cutout_offsets(xy_center, cutout_radius)
        xy_start_cutout = (xy_start[0] - cutout_x_offset, xy_start[1] - cutout_y_offset)
        xy_end_cutout = (xy_end[0] - cutout_x_offset, xy_end[1] - cutout_y_offset)

        foreground_mask = make_pill_mask(cutout_size, xy_start_cutout, xy_end_cutout, foreground_radius)
        radius_inside = self.foreground_radius + gap
        radius_outside = radius_inside + self.background_width
        background_mask_center_pill = np.logical_not(make_pill_mask(cutout_size, xy_start_cutout,
                                                                    xy_end_cutout, radius_inside))
        background_mask_outer_pill = make_pill_mask(cutout_size, xy_start_cutout,
                                                    xy_end_cutout, radius_outside)
        background_mask = np.logical_or(background_mask_center_pill, background_mask_outer_pill)
        super().__init__(parent, xy_center, cutout_radius, foreground_mask, background_mask)

    def make_new_object(self, new_xy_center):
        """ Overrides parent-class method.
            Make new object using new xy_center.
            Masks will be recreated by the constructor, using new xy_center.
        """
        x_half_motion, y_half_motion = self.xy_motion[0] / 2.0, self.xy_motion[1] / 2.0
        new_xy_start = new_xy_center[0] - x_half_motion, new_xy_center[1] - y_half_motion
        new_xy_end = new_xy_center[0] + x_half_motion, new_xy_center[1] + y_half_motion
        return Old_MovingSourceAp(self.parent, new_xy_start, new_xy_end,
                                  self.foreground_radius, self.gap, self.background_width)

