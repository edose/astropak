__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

""" Module astrosupport.image.
    Image, FITS, aperture handling.
    Forked from photrix.image 2020-10-23 EVD.
"""

# Python core packages:
import os
from datetime import datetime, timezone, timedelta
from math import floor, ceil, cos, sin, pi, sqrt, log

# External packages:
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import astropy.io.fits
from astropy.nddata import CCDData
from dateutil.parser import parse
from photutils import make_source_mask, centroid_com
from astropy.stats import sigma_clipped_stats

# EVD modules:
from .util import ra_as_degrees, dec_as_degrees, RaDec

TOP_DIRECTORY = 'C:/Astro/Images/Borea Photrix'
# FITS_REGEX_PATTERN = '^(.+)\.(f[A-Za-z]{2,3})$'
FITS_EXTENSIONS = ['fts', 'fit', 'fits']  # allowed filename extensions
ISO_8601_FORMAT = '%Y-%m-%dT%H:%M:%S'
# ISO_8601_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
FWHM_PER_SIGMA = 2.0 * sqrt(2.0 * log(2))
SUBIMAGE_MARGIN = 1.5  # subimage pixels around outer annulus, for safety
RADIANS_PER_DEGREE = pi / 180.0
DEGREES_PER_RADIAN = 180.0 / pi

# R_DISC altered 10 -> 9 Aug 16 2019 for new L-500 mount.
# This is here for safety only--normally, user would pass in values derived from some .ini file
#     with values specific to a specific photometry application.
R_DISC = 9  # for aperture photometry, likely to be adaptive (per image) later.
R_INNER = 15  # "
R_OUTER = 20  # "


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


class FITS:
    """ Holds data from a FITS file. Immutable. Used mostly by an Image object (class Image).
        Internally, ALL image data & coordinates are held as zero-based (y,x) arrays (python, first
            coordinate is y, second is x), and NOT as FITS which are (x,y), origin=1
        TESTS OK 2020-10-26.
    Usage: obj = FITS('C:/Astro', 'Borea', 'xxx.fts')  # 3 parts of full path.
    Usage: obj = FITS('C:/Astro/Borea/', 'xxx.fts')    # 2 parts of full path.
    Usage: obj = FITS('C:/Astro/Borea/xxx.fts')        # full path already available.
    """
    def __init__(self, top_directory, rel_directory=None, filename=None,
                 pinpoint_pixel_scale_multiplier=1):
        """
        :param top_directory:
        :param rel_directory:
        :param filename:
        :param pinpoint_pixel_scale_multiplier: value (prob. near 1) by which to multiply pixel scale
            IFF pinpoint plate solution is detected. The best solution I can devise for the PinPoint mess.
            Required because (sigh) Pinpoint plate solutions include "private" distortion parameters
            so that their WCS values are not what they would be & should be for a WCS-only solution.
            That is, the zero-order solution is *called* a proper WCS but is not, and it cannot be
            used as one, nor even correctable with its "private" distortion parameters.
        """
        # If filename has FITS extension already, use it:
        actual_fits_fullpath = None
        test_fullpath = top_directory
        for more in [rel_directory, filename]:
            if more is not None:
                test_fullpath = os.path.join(test_fullpath, more)
        if os.path.exists(test_fullpath):
            actual_fits_fullpath = test_fullpath

        # If no file with FITS extension, try to find a matching FITS filename (w/extension):
        if actual_fits_fullpath is None:
            for fits_ext in FITS_EXTENSIONS:
                test_fullpath = os.path.join(top_directory, rel_directory,
                                             filename + '.' + fits_ext)
                if os.path.exists(test_fullpath):
                    actual_fits_fullpath = test_fullpath
                break

        if actual_fits_fullpath is None:
            print("Not a valid file name: '" + filename + "'")
            self.is_valid = False
            return

        self.fullpath = actual_fits_fullpath
        try:
            hdulist = astropy.io.fits.open(self.fullpath)
        except IOError:
            self.is_valid = False
            return

        self.header = hdulist[0].header.copy()
        # self.header_keys = [key for key in self.header.keys()]      # save generator results as a list.
        # self.header_items = [item for item in self.header.items()]  # save generator results as a list.

        # FITS convention = (vert/Y, horiz/X), pixel (1,1) at bottom left -- NOT USED by photrix.
        # MaxIm/Astrometrica convention = (horiz/X, vert/Y) pixel (0,0 at top left). USE THIS.
        # NB: self.image_fits, self.image_xy, and self.image are different views of the SAME array.
        #     They are meant to be read-only--changing any one of them *will* change the others.
        self.image_fits = hdulist[0].data.astype(np.float64)
        self.image_xy = np.transpose(self.image_fits)  # x and y axes as expected (not like FITS).
        self.image = self.image_xy  # alias
        hdulist.close()

        self.top_directory = top_directory
        self.rel_directory = rel_directory
        self.filename = filename
        self.object = self.header_value('OBJECT')
        self.is_calibrated = self._is_calibrated()
        self.focal_length = self._get_focal_length()
        self.exposure = self.header_value(['EXPTIME', 'EXPOSURE'])  # seconds
        self.temperature = self.header_value(['SET-TEMP', 'CCD-TEMP'])  # deg C
        self.utc_start = self._get_utc_start()
        self.utc_mid = self.utc_start + timedelta(seconds=self.exposure / 2.0)
        self.filter = self.header_value('FILTER')
        self.airmass = self.header_value('AIRMASS')
        self.guide_exposure = self.header_value('TRAKTIME')  # seconds
        self.fwhm = self.header_value('FWHM')  # pixels

        # Note: self.plate_solution_is_pinpoint is needed before running .get_plate_solution().
        self.plate_solution_is_pinpoint = all([key in self.header.keys()
                                               for key in ['TR1_0', 'TR2_1', 'TR1_6', 'TR2_5']])
        self.plate_solution = self._get_plate_solution(pinpoint_pixel_scale_multiplier)  # a pd.Series
        self.is_plate_solved = not any(self.plate_solution.isnull())
        self.ra = ra_as_degrees(self.header_value(['RA', 'OBJCTRA']))
        self.dec = dec_as_degrees(self.header_value(['DEC', 'OBJCTDEC']))

        self.is_valid = True  # if it got through all that initialization.
        # self.is_valid = all(x is not None
        #                     for x in [self.object, self.exposure, self.filter,
        #                               self.airmass, self.utc_start, self.focal_length])

    def header_has_key(self, key):
        """ Return True iff key is in FITS header. No validation of value."""
        return key in self.header

    def header_value(self, key):
        """ Return value associated with given FITS header key.
        :param key: FITS header key [string] or list of keys to try [list of strings]
        :return: value of FITS header entry, typically [float] if possible, else [string]
        """
        if isinstance(key, str):
            return self.header.get(key, None)
        for k in key:
            value = self.header.get(k, None)
            if value is not None:
                return value
        return None

    def xy_from_radec(self, radec):
        """ Computes zero-based (python, not FITS) x and y for a given RA and Dec sky coordinate.
            May be outside image's actual boundaries.
            Assumes flat image (no distortion, i.e., pure Tan projection).
        :param radec: sky coordinates. [RaDec object]
        :return: x and y pixel position, zero-based, in this FITS image [2-tuple of floats]
        """
        cd11 = self.plate_solution['CD1_1']  # d(RA)/dx, deg/px
        cd12 = self.plate_solution['CD1_2']  # d(RA)/dy, deg/px
        cd21 = self.plate_solution['CD2_1']  # d(Dec)/dx, deg/px
        cd22 = self.plate_solution['CD2_2']  # d(Dec)/dy, deg/px
        crval1 = self.plate_solution['CRVAL1']  # center RA in degrees
        crval2 = self.plate_solution['CRVAL2']  # center Dec in degrees
        crpix1 = self.plate_solution['CRPIX1']  # 1 at edge (FITS convention)
        crpix2 = self.plate_solution['CRPIX2']  # "

        d_ra = radec.ra - crval1
        d_dec = radec.dec - crval2
        deg_ew = d_ra * cos((pi / 180.0) * radec.dec)
        deg_ns = d_dec
        a = cd22 / cd12
        dx = (deg_ns - deg_ew * a) / (cd21 - cd11 * a)
        dy = (deg_ew - cd11 * dx) / cd12
        x = crpix1 + dx
        y = crpix2 + dy
        return x - 1, y - 1  # FITS image origin=(1,1), but our (MaxIm/python) convention=(0,0)

    def radec_from_xy(self, x, y):
        """ Calculate RA, Dec for given (x,y) pixel position in this image. Uses WCS (linear) plate soln.
        :param x: pixel position in X. [float}
        :param y: pixel position in Y. [float]
        :return: RA,Dec sky position for given pixel position. [RaDec object]
        """
        ps = self.plate_solution
        # TODO: center pixel should probably be read directly from plate solution.
        dx = x - self.image.shape[0] / 2.0
        dy = y - self.image.shape[1] / 2.0
        d_east_west = (dx * ps['CD1_1'] + dy * ps['CD1_2'])
        d_ra = d_east_west / cos(ps['CRVAL2'] / DEGREES_PER_RADIAN)
        d_dec = (dx * ps['CD2_1'] + dy * ps['CD2_2'])
        ra = ps['CRVAL1'] + d_ra
        dec = ps['CRVAL2'] + d_dec
        return RaDec(ra, dec)

    def bounding_ra_dec(self, extension_percent=3):
        """ Returns bounding RA and Dec that will completely cover this plate-solved FITS image.
        :param extension_percent: to extend bounding box by x% beyond actual edges, enter x. [float]
        :return: min RA, max RA, min Dec, max Dec, all in degrees. [4-tuple of floats]
        """
        image = self.image_fits
        xsize, ysize = image.shape[1], image.shape[0]
        ps = self.plate_solution  # a pandas Series
        ra_list, dec_list = [], []
        extension_fraction = extension_percent / 100.0
        for x in [-extension_fraction * xsize, (1 + extension_fraction) * xsize]:
            for y in [-extension_fraction * ysize, (1 + extension_fraction) * ysize]:
                radec = self.radec_from_xy(x, y)
                ra_list.append(radec.ra)
                dec_list.append(radec.dec)
        ra_deg_min = min(ra_list) % 360.0
        ra_deg_max = max(ra_list) % 360.0
        dec_deg_min = min(dec_list)
        dec_deg_max = max(dec_list)
        return ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max

    def _is_calibrated(self):
        calib_fn_list = [self._is_calibrated_by_maxim_5_6()]  # may add more fns when available.
        return any([is_c for is_c in calib_fn_list])

    def _is_calibrated_by_maxim_5_6(self):
        hval = self.header_value('CALSTAT')
        if hval is not None:
            if hval.strip().upper() == 'BDF':  # calib. by MaxIm DL v. 5 or 6
                return True
        return False

    def _get_focal_length(self):
        # If FL available, return it. Else, compute FL from plate solution.
        value = self.header_value('FOCALLEN')
        if value is not None:
            return value  # mm
        x_pixel = self.header_value('XPIXSZ')
        y_pixel = self.header_value('YPIXSZ')
        x_scale = self.header_value('CDELT1')
        y_scale = self.header_value('CDELT2')
        if any([val is None for val in [x_pixel, y_pixel, x_scale, y_scale]]):
            return None
        fl_x = x_pixel / abs(x_scale) * (206265.0 / (3600 * 1800))
        fl_y = y_pixel / abs(y_scale) * (206265.0 / (3600 * 1800))
        return (fl_x + fl_y) / 2.0

    def _get_utc_start(self):
        utc_string = self.header_value('DATE-OBS')
        # dateutil.parse.parse handles MaxIm 6.21 inconsistent format; datetime.strptime() can fail.
        utc_dt = parse(utc_string).replace(tzinfo=timezone.utc)
        return utc_dt

    def _get_plate_solution(self, pinpoint_pixel_scale_multiplier=1):
        """ Get plate solution's (WCS) 8 values, then apply pixel scale multipler to correct if needed.
        :param pinpoint_pixel_scale_multiplier: value (prob. near 1) by which to multiply pixel scale
            IFF pinpoint plate solution is detected. The best solution I can devise for the PinPoint mess.
            Required because (sigh) Pinpoint plate solutions include "private" distortion parameters
            so that their WCS values are not what they would be & should be for a WCS-only solution.
            That is, the zero-order solution is *called* a proper WCS but is not, and it cannot be
            used as one, nor even correctable with its "private" distortion parameters.
        :return: the 8 WCS values. [dict of 8 string:float items]
        """
        plate_solution_index = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                                'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
        plate_solution_values = [np.float64(self.header_value(key))
                                 for key in plate_solution_index]
        solution = pd.Series(plate_solution_values, index=plate_solution_index)
        # If CD terms are absent, and if plate solution resides in other terms
        # (e.g., in WCS from Astrometrica), then try to generate CD terms from other plate solution terms:
        if np.isnan(solution['CD1_1']):
            if self.header_value('CDELT1') is not None and self.header_value('CROTA2') is not None:
                solution['CD1_1'] = self.header_value('CDELT1') * \
                                    cos(self.header_value('CROTA2') * RADIANS_PER_DEGREE)
        if np.isnan(solution['CD1_2']):
            if self.header_value('CDELT2') is not None and self.header_value('CROTA2') is not None:
                solution['CD1_2'] = - self.header_value('CDELT2') * \
                                    sin(self.header_value('CROTA2') * RADIANS_PER_DEGREE)
        if np.isnan(solution['CD2_1']):
            if self.header_value('CDELT1') is not None and self.header_value('CROTA2') is not None:
                solution['CD2_1'] = self.header_value('CDELT1') * \
                                    sin(self.header_value('CROTA2') * RADIANS_PER_DEGREE)
        if np.isnan(solution['CD2_2']):
            if self.header_value('CDELT2') is not None and self.header_value('CROTA2') is not None:
                solution['CD2_2'] = self.header_value('CDELT2') * \
                                    cos(self.header_value('CROTA2') * RADIANS_PER_DEGREE)
        if self.plate_solution_is_pinpoint:
            for key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                solution[key] *= pinpoint_pixel_scale_multiplier
        return solution


class Cutout:
    """ Copy of image slice. Includes foreground and background masks and recentroiding.
        Much more versatile successor to mp_phot.util's class Square.
        IMMUTABLE once constructed; recentroiding generates new Cutout object(s).
        Mask convention: True = MASKED out & thus unused, False = unmasked & thus valid and used
            (same as for astropy's CCDData objects & numpy's masked arrays).
    """
    def __init__(self, parent, xy_center, cutout_radius, foreground_mask=None, background_mask=None):
        """ General constructor.
        :param parent: parent array [CCDData, numpy masked array, or numpy ndarray].
        :param xy_center: center pixel position in parent. [2-tuple or 2-array or 2-list, of floats]
        :param cutout_radius: half-length of cutout's edge length less one, in pixels.
                   Actual size of cutout will be from cutout_radius rounded up,
                   that is, from 2 * ceil(cutout_radius) + 1.
                   So size and shape are always an odd number of pixels, no exceptions. [float]
        :param foreground_mask: mask array for pixels to be counted in flux, centroid, etc.
                   Shape must exactly match shape of either the parent array or the cutout array.
                   Array True -> MASKED pixel, False -> pixel is valid and used. (numpy convention).
                   None means 'use all pixels'. [numpy ndarray of booleans, or None]
        :param background_mask: mask array for pixels to be counted in background flux, centroid, etc.
                   Shape must exactly match shape of either the parent array or the cutout array.
                   Array True -> MASKED pixel, False -> pixel is valid and used. (numpy convention).
                   None means background is assumed zero (has no effect).
                   [numpy ndarray of booleans, or None]
        """
        self.parent = parent
        self.xy_center = tuple(xy_center)
        self.input_radius = cutout_radius
        self.input_foreground_mask = foreground_mask
        self.input_background_mask = background_mask
        self.messages = []  # warnings and/or errors for this object [list of strings].
        input_messages = self._input_problems_detected()
        if len(input_messages) != 0:
            self.messages.extend(input_messages)
            self.is_valid = False
            return

        # Determine raw cutout boundaries, in parent pixels (cutout shape will be (odd int, odd int)):
        self.x_cutout_center = int(round(self.xy_center[0]))  # center cutout on nearest whole parent pixel.
        self.y_cutout_center = int(round(self.xy_center[1]))  # "
        self.radius = int(ceil(cutout_radius))  # round radius UP, if a float.
        self.x_raw_low = self.x_cutout_center - self.radius   # parent x-pixel of cutout zero x-pixel.
        self.x_raw_high = self.x_cutout_center + self.radius  # parent x-pixel of cutout highest x-pixel.
        self.y_raw_low = self.y_cutout_center - self.radius   # parent y-pixel of cutout zero y-pixel.
        self.y_raw_high = self.y_cutout_center + self.radius  # parent y-pixel of cutout highest y-pixel.
        self.x_offset = self.x_raw_low  # alias name
        self.y_offset = self.y_raw_low  # "
        edge_pixel_count = self.radius * 2 + 1
        self.shape = (edge_pixel_count, edge_pixel_count)
        x_data_low = max(0, self.x_raw_low)
        x_data_high = min(parent.shape[1] - 1, self.x_raw_high)
        y_data_low = max(0, self.y_raw_low)
        y_data_high = min(parent.shape[0] - 1, self.y_raw_high)
        if (x_data_low > x_data_high) or (y_data_low > y_data_high):
            self.messages.append('Cutout boundaries are outside parent array. '
                                 'Cutout object cannot be constructed.')
            self.is_valid = False
            return

        # Extract parent data array:
        if isinstance(self.parent, CCDData):
            self.parent_data = parent.data.copy()
        elif isinstance(self.parent, np.ma.core.MaskedArray):
            self.parent_data = parent.data.copy()
        elif isinstance(self.parent, np.ndarray):
            self.parent_data = parent.copy()
        else:
            self.is_valid = False  # should never get here if ._input_problems_detected() was run.
            return

        # Make cutout data array; fill with np.nan any pixels outside parent:
        self.data = np.full(self.shape, fill_value=np.nan, dtype=np.double)  # most will be overwritten.
        parent_data_for_cutout = self.parent_data[y_data_low: y_data_high + 1,
                                                  x_data_low: x_data_high + 1].copy()
        self.data[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                  x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = parent_data_for_cutout

        # Mask (invalidate) all cutout pixels outside parent:
        cutout_invalid_mask = np.full_like(self.data, fill_value=True, dtype=np.bool)  # all invalid.
        cutout_valid_mask = np.full_like(self.data, fill_value=False, dtype=np.bool)  # all valid.
        within_parent_mask = cutout_invalid_mask.copy()
        within_parent_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                           x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = False
        self.is_all_within_parent = not np.any(within_parent_mask)  # NB: False = valid (within parent).
        if not self.is_all_within_parent:
            self.is_pristine = False

        # Make foreground mask. If explicitly given, use that only. A given mask must be either
        #     of parent's shape (make a cutout of it), or of cutout's own shape (use it directly).
        #     Any other mask shape causes a failure.
        # If mask not given (is None): use mask from input object (CCDData or numpy masked array), or for
        #     data-only numpy ndarray given, set mask to make valid the entire cutout.
        # Finally, always mask out any cutout pixels outside of parent data array.
        if self.input_foreground_mask is not None:
            if self.input_foreground_mask.shape == self.parent.shape:
                # Case: extract mask data from parent-shaped mask array:
                raw_foreground_mask = cutout_invalid_mask.copy()
                raw_foreground_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                                    x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    self.input_foreground_mask[y_data_low: y_data_high + 1,
                                               x_data_low: x_data_high + 1].copy()
            elif self.input_foreground_mask.shape == self.shape:
                # Case: copy mask data from cutout-shaped mask:
                raw_foreground_mask = cutout_invalid_mask.copy()
                raw_foreground_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                                    x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    self.input_foreground_mask[y_data_low - self.y_offset:
                                               (y_data_high + 1) - self.y_offset,
                                               x_data_low - self.x_offset:
                                               (x_data_high + 1) - self.x_offset]
            else:
                self.messages.append('Input foreground mask shape matches neither parent nor cutout shape.')
                self.is_valid = False
                return
            self.foreground_mask = np.logical_or(raw_foreground_mask, within_parent_mask)
        else:
            if isinstance(self.parent, CCDData) or isinstance(self.parent, np.ma.core.MaskedArray):
                raw_foreground_mask = cutout_invalid_mask
                extracted_foreground_mask = self.parent.mask[y_data_low: y_data_high + 1,
                                                             x_data_low: x_data_high + 1].copy()
                raw_foreground_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                                    x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    extracted_foreground_mask
            elif isinstance(self.parent, np.ndarray):
                raw_foreground_mask = np.full_like(self.data, fill_value=False, dtype=np.bool)  # all valid.
            else:
                self.messages.append('Parent type not in approved list.')
                self.is_valid = False
                return
            self.foreground_mask = np.logical_or(raw_foreground_mask, within_parent_mask)

        # Make background mask. If explicitly given, use that only. A given mask must be either
        #     of parent's shape (make a cutout of it), or of cutout's own shape (use it directly).
        #     Any other mask shape causes a failure.
        # If mask not given (is None): use inverse of foreground mask.
        # Finally, always mask out any cutout pixels outside of parent data array.
        if self.input_background_mask is not None:
            if self.input_background_mask.shape == self.parent.shape:
                # Case: extract mask data from parent-shaped mask array:
                raw_background_mask = cutout_invalid_mask.copy()
                raw_background_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                                    x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    self.input_background_mask[y_data_low: y_data_high + 1,
                                               x_data_low: x_data_high + 1].copy()
            elif self.input_background_mask.shape == self.shape:
                # Case: copy mask data from cutout-shaped mask:
                raw_background_mask = cutout_invalid_mask.copy()
                raw_background_mask[y_data_low - self.y_offset: (y_data_high + 1) - self.y_offset,
                                    x_data_low - self.x_offset: (x_data_high + 1) - self.x_offset] = \
                    self.input_background_mask[y_data_low - self.y_offset:
                                               (y_data_high + 1) - self.y_offset,
                                               x_data_low - self.x_offset:
                                               (x_data_high + 1) - self.x_offset]
            else:
                self.messages.append('Input background mask shape matches neither parent nor cutout shape.')
                self.is_valid = False
                return
            self.background_mask = np.logical_or(raw_background_mask, within_parent_mask)
        else:
            raw_background_mask = np.logical_not(self.foreground_mask)  # simply inverse of foreground mask.
            self.background_mask = np.logical_or(raw_background_mask, within_parent_mask)

        # Pixel counts:
        self.pixel_count = self.shape[0] * self.shape[1]
        self.foreground_pixel_count = np.sum(self.foreground_mask == False)
        self.background_pixel_count = np.sum(self.background_mask == False)

        # Warn if any overlap between foreground and background masks (usually indicates a problem):
        self.mask_overlap_pixel_count = np.sum(np.logical_not(np.logical_or(self.foreground_mask,
                                                                            self.background_mask)))
        if np.sum(self.mask_overlap_pixel_count) > 0:
            self.messages.append('Warning: ' + str(self.mask_overlap_pixel_count) +
                                 ' pixels are in both foreground and background masks.')
            self.is_pristine = False

        self.xy_centroid = self._calc_centroid()  # 2-tuple of floats
        self.is_valid = True
        self.is_pristine = (len(self.messages) == 0) and self.is_valid

    def __str__(self):
        return 'Cutout object of x,y shape (' + str(self.shape[1]) + ', ' + str(self.shape[0]) + ')' +\
               ' from parent of x,y shape (' + str(self.parent.shape[1]) + ', ' + \
               str(self.parent.shape[0]) + ')'

    def _input_problems_detected(self):
        messages = []
        if not type(self.parent) in [CCDData, np.ma, np.ndarray]:
            messages.append('Input array \'parent\' is of invalid type.')
        if len(self.xy_center) != 2:
            messages.append('Input tuple \'xy_center\' is invalid, must be 2-tuple of floats or ints.')
        elif (self.xy_center[0] < 0) or (self.xy_center[0] > self.parent.shape[1] - 1) or \
             (self.xy_center[1] < 0) or (self.xy_center[1] > self.parent.shape[0] - 1):
            messages.append('Input \'xy_center\' is outside parent array.')
        if self.input_radius <= 0:
            messages.append('Input \'input radius\' invalid, must be positive int or float.')
        if self.input_foreground_mask is not None:
            invalidity_string = 'Input \'foreground_mask\' invalid, must be a numpy ndarray of booleans.'
            if type(self.input_foreground_mask) != np.ndarray:
                messages.append(invalidity_string)
            elif self.input_foreground_mask.dtype != np.bool:
                messages.append(invalidity_string)
        if self.input_background_mask is not None:
            invalidity_string = 'Input \'background_mask\' invalid, must be a numpy ndarray of booleans.'
            if type(self.input_background_mask) != np.ndarray:
                messages.append(invalidity_string)
            elif self.input_background_mask.dtype != np.bool:
                messages.append(invalidity_string)
        return messages

    @classmethod
    def circular_mask_and_radius(cls, parent, xy_center, cutout_radius, foreground_mask_radius, gap=0.0):
        """ Alternate constructor, typically for stationary light sources (like stars) within an image.
            Circular mask centered in cutout, with explicit radius.
        :param parent: parent array [CCDData, numpy masked array, or numpy ndarray].
        :param xy_center: center pixel position in parent. [2-tuple or 2-array or 2-list, of floats]
        :param cutout_radius: half-length of cutout's edge length less one, in pixels. [float]
        :param foreground_mask_radius: radius of foreground (star) mask, in pixels [float]
        :param gap: gap between outer edge of foreground mask and inner edge of background mask,
                   in pixels. Useful for aperture photometry of stars. Leave at zero for no gap. [float]
        :return: Cutout object as defined.
        """
        raw_cutout = Cutout(parent, xy_center, cutout_radius, foreground_mask=None, background_mask=None)
        x0 = raw_cutout.xy_center[0] - raw_cutout.x_offset
        y0 = raw_cutout.xy_center[1] - raw_cutout.y_offset
        new_foreground_mask = make_circular_mask(raw_cutout.shape[0], (x0, y0), foreground_mask_radius)
        if gap <= 0.0:
            # No gap; let Cutout constructor make default background mask.
            return Cutout(parent, xy_center, cutout_radius, foreground_mask=new_foreground_mask)
        else:
            background_mask_radius = foreground_mask_radius + gap
            new_background_mask = np.logical_not(make_circular_mask(raw_cutout.shape[0], (x0, y0),
                                                                    background_mask_radius))
            return Cutout(parent, xy_center, cutout_radius, foreground_mask=new_foreground_mask,
                          background_mask=new_background_mask)

    @classmethod
    def circular_mask_and_margin(cls, parent, xy_center, margin, foreground_mask_radius, gap=0.0):
        """ Alternate constructor, typically for stationary light sources (like stars) within an image.
            Circular mask centered in cutout, with cutout's size the smallest that allows for
                the foreground mask, a gap to the (circular) background mask, and a margin.
        :param parent: parent array [CCDData, numpy masked array, or numpy ndarray].
        :param xy_center: center pixel position in parent. [2-tuple or 2-array or 2-list, of floats]
        :param margin: pixel span required between mask edge & its gap to the cutout edge.
        :param foreground_mask_radius: radius of foreground (star) mask, in pixels [float]
        :param gap: gap between outer edge of foreground mask and inner edge of background mask,
                   in pixels. Useful for aperture photometry. Leave at zero for no gap. [float]
        :return: Cutout object as defined.
        """
        cutout_radius = int(ceil(foreground_mask_radius + gap + margin))
        return Cutout.circular_mask_and_radius(parent, xy_center, cutout_radius,
                                               foreground_mask_radius, gap)

    @classmethod
    def pill_mask_and_radius(cls, parent, xy_start, xy_end, cutout_radius, foreground_mask_radius, gap=0.0):
        """ Alternate constructor, typically for moving light sources (like MPs) within an image.
            Pill-shaped mask: a line segment between xy_start and xy_end,
                expanded circularly by mask_radius. Specify cutout_radius.
        :param parent: parent array [CCDData, numpy masked array, or numpy ndarray].
        :param xy_start:
        :param xy_end:
        :param cutout_radius: half-length of cutout's edge length less one, in pixels. [float]
        :param foreground_mask_radius: radius of foreground (star) mask, in pixels [float]
        :param gap: gap between outer edge of foreground mask and inner edge of background mask,
                   in pixels. Useful for aperture photometry of stars. Leave at zero for no gap. [float]
        :return: Cutout object as defined.
        """
        # Center cutout on the center of moving source's track:
        new_cutout_radius = int(ceil(cutout_radius))  # next higher integer.
        cutout_size = int(ceil(new_cutout_radius * 2 + 1))  # size must be an odd integer.
        xy_center = (xy_start[0] + xy_end[0]) / 2.0, (xy_start[1] + xy_end[1]) / 2.0
        raw_cutout = Cutout(parent, xy_center, cutout_radius, foreground_mask=None, background_mask=None)
        x_start_cutout = xy_start[0] - raw_cutout.x_offset
        y_start_cutout = xy_start[1] - raw_cutout.y_offset
        x_end_cutout = xy_end[0] - raw_cutout.x_offset
        y_end_cutout = xy_end[1] - raw_cutout.y_offset
        xy_start_cutout = (x_start_cutout, y_start_cutout)
        xy_end_cutout = (x_end_cutout, y_end_cutout)
        new_foreground_mask = make_pill_mask(cutout_size, xy_start_cutout, xy_end_cutout,
                                             foreground_mask_radius)
        if gap > 0.0:
            background_mask_radius = foreground_mask_radius + gap
            new_background_mask = np.logical_not(make_pill_mask(cutout_size, xy_start_cutout, xy_end_cutout,
                                                                background_mask_radius))
            return Cutout(parent, xy_center, cutout_radius, new_foreground_mask, new_background_mask)
        else:
            # Let Cutout's constructor make background mask from inverse of foreground mask.
            return Cutout(parent, xy_center, cutout_radius, new_foreground_mask)

    @classmethod
    def pill_mask_and_margin(cls, parent, xy_start, xy_end, margin, foreground_mask_radius, gap=0.0):
        """ Alternate constructor, typically for moving light sources (like MPs) within an image.
            Pill-shaped mask: a line segment between xy_start and xy_end,
                expanded circularly by mask_radius. Specify mask_radius, which sets cutout_radius to fit.
        :param parent: parent array [CCDData, numpy masked array, or numpy ndarray].
        :param xy_start:
        :param xy_end:
        :param margin: pixel span required between mask edge and cutout edge.
        :param foreground_mask_radius: radius of foreground (minor planet) mask, in pixels [float]
        :param gap: gap between outer edge of foreground mask and inner edge of background mask,
                   in pixels. Useful for aperture photometry. Leave at zero for no gap. [float]
        :return: Cutout object as defined.
        """
        foreground_mask_x_span = abs(xy_start[0] - xy_end[0]) + 2 * foreground_mask_radius
        foreground_mask_y_span = abs(xy_start[1] - xy_end[1]) + 2 * foreground_mask_radius
        max_foreground_mask_span = max(foreground_mask_x_span, foreground_mask_y_span)
        cutout_radius = int(ceil(max_foreground_mask_span / 2 + margin + gap))
        return Cutout.pill_mask_and_radius(parent, xy_start, xy_end, cutout_radius,
                                           foreground_mask_radius, gap)

    def flux(self, use_foreground_mask=True, use_background_mask=True, gain=1):
        """ Return flux (foreground minus background) of current object, using masks iff desired.
        If foreground mask is used, all unmasked (valid) pixels are summed.
        If foreground mask not used, all pixels are summed for foreground flux.
        If background mask is used and has valid pixels, sigma-clipped median level is subtracted
            from each foreground pixel before summing foreground.
        If background mask not used, background flux is set to zero (to have no effect).
        :param use_foreground_mask: [boolean]
        :param use_background_mask: [boolean]
        :param gain: CCD-like gain in e-/ADU. [float]
        :return: background-adjusted foreground (source) 'net' flux,
                 std dev uncertainty of net flux,
                 background level (sigma-clipped median), and
                 std dev uncertainty in background level (per pixel, not of background area's average).
                 [4-tuple of floats]
        """
        if use_foreground_mask is True:
            raw_flux = np.sum(self.data, where=np.logical_not(self.foreground_mask))
        else:
            raw_flux = np.sum(self.data)
        flux_variance_from_poisson = raw_flux / gain  # from var(e-) = flux in e-.

        if use_background_mask is True and self.background_pixel_count >= 2:
            background_level, background_stddev = calc_background_value(self.data, self.background_mask)
        else:
            background_level, background_stddev = 0.0, 0.0  # no effect on flux or stddev.

        background_adjusted_flux = raw_flux - self.foreground_pixel_count * background_level

        flux_variance_from_background = self.foreground_pixel_count * \
                                        ((background_stddev ** 2) / self.background_pixel_count)
        flux_variance = flux_variance_from_poisson + flux_variance_from_background
        flux_stddev = sqrt(flux_variance)
        return background_adjusted_flux, flux_stddev, background_level, background_stddev

    def _calc_centroid(self):
        """ Return position (parent image) of local flux centroid within cutout.
            Always background-subtracted. No iteration.
        :return: position (x,y in *parent* image) of local flux centroid. [2-tuple of floats]"""
        y_grid, x_grid = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        background_level, _ = calc_background_value(self.data, self.background_mask)
        data_minus_background = self.data - background_level
        x_net_product = x_grid * data_minus_background
        y_net_product = y_grid * data_minus_background
        x_product_sum = np.sum(x_net_product, where=np.logical_not(self.foreground_mask))
        y_product_sum = np.sum(y_net_product, where=np.logical_not(self.foreground_mask))
        data_sum = np.sum(data_minus_background, where=np.logical_not(self.foreground_mask))
        x_centroid = self.x_offset + (x_product_sum / data_sum)
        y_centroid = self.y_offset + (y_product_sum / data_sum)
        return x_centroid, y_centroid








    def refine_centroid(self, max_adjustment=None, max_iterations=3):
        """ Returns new Cutout object with refined centroid.
            NB: New Cutout object may be displaced from present object by a pixel in either x or y.
        :param max_adjustment: will terminate iterations if centroid adjustment is smaller than this value,
                               in pixels. [float]
        :param max_iterations: will terminate after this number of iterations in any case. [int]
        :return: new Cutout object with refined centroid. [Cutout object]
        """
        pass  # TODO: As from mp_phot.Square, but may not be the right way to do this.



_____IMAGE_and_GEOMETRY_SUPPORT____________________________________ = 0


def make_circular_mask(mask_size, xy, radius):
    """ Construct a traditional mask array for small, stationary object, esp. for a star.
        Unmask only those pixels *within* radius pixels of a given point. Invert the mask separately to
            mask the interior. Convention: pixel True -> VALID (opposite of numpy).
    :param mask_size: edge size of new mask array, should be odd number of pixels. [int]
    :param xy: (x, y) pixel coordinates of central point. [2-tuple of floats]
    :param radius: radius of ends and half-width of center region. [float]
    :return: mask array, True -> VALID (opposite convention from numpy). [np.ndarray of booleans]
    """
    x0, y0 = xy
    new_mask = np.fromfunction(lambda i, j: ((j - x0) ** 2 + (i - y0) ** 2) > radius ** 2,
                               shape=(mask_size, mask_size))  # nb: True masks out, as numpy.
    return new_mask


def make_pill_mask(mask_size, xya, xyb, radius):
    """ Construct a mask array for MP in motion: unmask only those pixels within radius pixels of
        any point in line segment from xya to xyb. Convention: pixel True -> VALID (opposite of numpy).
    :param mask_size: edge size of new mask array, should be odd number of pixels. [int]
    :param xya: (xa, ya) pixel coordinates of start-motion point. [2-tuple of floats]
    :param xyb: (xb, yb) pixel coordinates of end-motion point. [2-tuple of floats]
    :param radius: radius of ends and half-width of center region. [float]
    :return: mask array, True -> VALID (opposite convention from numpy). [np.ndarray of booleans]
    """
    xa, ya = tuple(xya)
    xb, yb = tuple(xyb)
    dx = xb - xa
    dy = yb - ya
    distance_motion = sqrt(dx**2 + dy**2)

    # Unmask up to radius distance from each endpoint:
    circle_a_mask = np.fromfunction(lambda i, j: ((i - ya) ** 2 + (j - xa) ** 2) > (radius ** 2),
                                    shape=(mask_size, mask_size))
    circle_b_mask = np.fromfunction(lambda i, j: ((i - yb) ** 2 + (j - xb) ** 2) > (radius ** 2),
                                    shape=(mask_size, mask_size))

    # Mask outside max distance from (xa,ya)-(xb,yb) line segment:
    rectangle_submask_1 = np.fromfunction(lambda i, j:
                                          distance_to_line((j, i), (xa, ya), (xb, yb), distance_motion) >
                                          radius, shape=(mask_size, mask_size))

    # Mask ahead of or behind MP motion line segment:
    dx_left = dy
    dy_left = -dx
    dx_right = -dy
    dy_right = dx
    x_center = (xa + xb) / 2.0
    y_center = (ya + yb) / 2.0
    x_left = x_center + dx_left
    y_left = y_center + dy_left
    x_right = x_center + dx_right
    y_right = y_center + dy_right
    distance_perpendicular = sqrt((x_right - x_left)**2 + (y_right - y_left)**2)  # prob = distance_motion
    rectangle_submask_2 = np.fromfunction(lambda i, j:
                                          distance_to_line((j, i), (x_left, y_left), (x_right, y_right),
                                                           distance_perpendicular) > distance_motion / 2.0,
                                          shape=(mask_size, mask_size))
    # Combine masks and return:
    rectangle_mask = np.logical_or(rectangle_submask_1, rectangle_submask_2)  # intersection of False.
    circles_mask = np.logical_and(circle_a_mask, circle_b_mask)  # union of False.
    mask = np.logical_and(circles_mask, rectangle_mask)          # "
    return mask


def distance_to_line(xy_pt, xy_a, xy_b, dist_ab=None):
    """ Yield the closest (perpendicular) distance from point (xpt, ypt) to the line (not necessarily
        within the closed line segment) passing through (x1,y1) and (x2,y2). """
    xpt, ypt = tuple(xy_pt)
    xa, ya = tuple(xy_a)
    xb, yb = tuple(xy_b)
    if dist_ab is None:
        dist_ab = sqrt((yb - ya)**2 + (xb - xa)**2)
    distance = abs((yb - ya) * xpt - (xb - xa) * ypt + xb * ya - yb * xa) / dist_ab
    return distance


def calc_background_value(data, mask=None):
    """ Calculate the sigma-clipped median value of a (possibly masked) data array.
    :param data: array of pixels. [2-D ndarray of floats]
    :param mask: mask array, True=masked, i.e., use only False pixels. [None, or 2-D nadarray of bool]
    :return: tuple of background adu level (flux per pixel), standard deviation within used pixels.
                 [2-tuple of floats]
    """
    if mask is None:  # use all pixels.
        this_mask = np.full_like(data, False, dtype=np.float)
    elif mask.shape != data.shape:  # bad mask shape.
        return None
    elif np.sum(mask == False) == 0:  # no valid pixels.
        return 0.0, 0.0
    else:
        this_mask = mask.copy()

    source_mask = make_source_mask(data, mask=this_mask, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, std = sigma_clipped_stats(data, sigma=3.0, mask=source_mask)
    return median, std



_____FITS_FILE_HANDLING___________________________________ = 0

def all_fits_files(top_directory, rel_directory, validate_fits=False):
    """  Return list of all FITS files in given directory_path.
         (Code for this exists already, somewhere.)
    :param top_directory:
    :param rel_directory:
    :param validate_fits: If True, open FITS files and include only if valid.
        If False, include filename if it appears valid without opening the FITS file.
    :return: List of all FITS files in given directory_path [list of strings]
    """
    # TODO: write all_fits_files().
    pass


def bounding_box_all_fits_files(top_directory, rel_directory):
    """ Return a RA,Dec bounding box that covers *all* FITS files in a given directory.
        (Calls all_fits_files(), then iterates through files with FITS.bounding_ra_dec().)
    :param top_directory:
    :param rel_directory:
    :return:
    """
    # TODO: write bounding_box_all_fits_files().
    pass
