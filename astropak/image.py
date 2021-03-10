__author__ = "Eric Dose :: Albuquerque"

""" Module astrosupport.image.
    Image, FITS, aperture handling.
    Forked from photrix.image 2020-10-23 EVD.
"""

# Python core packages:
import os
from datetime import timezone, timedelta
from math import ceil, cos, sin, pi, sqrt, log

# External packages:
import numpy as np
import pandas as pd
import astropy.io.fits
from dateutil.parser import parse
from photutils import make_source_mask, data_properties
from astropy.stats import sigma_clipped_stats

# Author's packages:
from astropak.util import ra_as_degrees, dec_as_degrees, RaDec
from astropak.reference import RADIANS_PER_DEGREE, DEGREES_PER_RADIAN, FWHM_PER_SIGMA
from astropak.geometry import XY, DXY, Circle_in_2D, Rectangle_in_2D


TOP_DIRECTORY = 'C:/Astro/Images/Borea Photrix'
# FITS_REGEX_PATTERN = '^(.+)\.(f[A-Za-z]{2,3})$'
FITS_EXTENSIONS = ['fts', 'fit', 'fits']  # allowed filename extensions
ISO_8601_FORMAT = '%Y-%m-%dT%H:%M:%S'


_____LEGACY_CLASS_from_package_PHOTRIX_______________________________ = 0


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
                ra_list.append(radec.ra % 360)  # ensure within [0, 360).
                dec_list.append(radec.dec)

        # Special RA algorithm to handle corners straddling RA=0:
        ra_deg_min, ra_deg_max = calc_ra_limits(ra_list)
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


_____AP_Parent_Class_and_working_subclasses________________________________________ = 0


class Ap:
    """ Parent class of all apertures for aperture photometry, each shape being a specific subclass.
        Defines a slice of a full image based on parameters passed in, computes and holds properties.
        Ap-class objects are IMMUTABLE once constructed; recenterin generates and returns a new object.
        Masks are boolean arrays, must match data in shape, and follow numpy's mask convention:
            mask pixel=True means corresponding data pixel is invalid ("masked away") and not used.
            mask pixel=False means data pixel is valid and used.
        Masks are foreground (light source) and background (surrounding sky).
        Any specified pixels that would fall outside parent image are excluded (cutout is cropped).
    """
    def __init__(self, image, xy_center, xy_origin, foreground_mask, background_mask=None):
        """ General constructor, from explicitly passed-in parent data array and 2 mask arrays.
        :param image: the parent image array [numpy ndarray; to pass in CCDData or numpy masked array,
                   please see separate, specific constructors, below].
        :param xy_center: center pixel position in parent. This should be the best prior estimate
                   of the light source's centroid at mid-exposure, as (x,y) (not as numpy [y, x] array).
                   [XY object, 2-tuple, 2-list, or 2-array of floats]
        :param xy_origin: lowest index of cutout (upper-left corner of image).
                   [XY object, 2-tuple, 2-list, or 2-array of floats]
        :param foreground_mask: mask array for pixels to be counted in flux, centroid, etc.
                   Required boolean array, numpy convention (True -> pixel masked out, unused).
                   Mask shape defines shape of cutout to be used. [numpy ndarray of booleans]
        :param background_mask: mask array for pixels to be counted in background flux, centroid, etc.
                   If not None, shape must exactly match shape of foreground_mask.
                   Optional boolean array, numpy convention (True -> pixel masked out, unused).
                   Specifying None means background unusused (zero). [numpy ndarray of booleans, or None]
        """
        self.image = image
        self.xy_center = xy_center if isinstance(xy_center, XY) else XY(xy_center[0], xy_center[1])
        self.xy_input_origin = xy_origin if isinstance(xy_origin, XY) else XY(xy_origin[0], xy_origin[1])
        self.input_foreground_mask = foreground_mask
        self.input_background_mask = background_mask
        self.input_shape = self.input_background_mask.shape
        self.dxy_input_shape = DXY(self.input_shape[1], self.input_shape[0])
        self.messages = []

        # Crop cutout to fit inside image:
        self.xy_origin = XY(max(0, self.xy_input_origin.x), max(0, self.xy_input_origin.y))
        self.xy_max = XY(min(image.shape[1] - 1, self.xy_input_origin.x + self.input_shape[1]),
                         min(image.shape[0] - 1, self.xy_input_origin.y + self.input_shape[0]))
        self.dxy_shape = self.xy_max - self.xy_origin + DXY(1, 1)
        self.cutout = self.image[self.xy_origin.x:self.xy_max.x + 1, self.xy_origin.y:self.xy_max.y + 1]
        must_crop = self.dxy_shape != self.dxy_input_shape
        if must_crop:
            dxy_to_crop_low = self.xy_origin - self.xy_input_origin
            foreground_mask_cropped_low = self.input_foreground_mask[dxy_to_crop_low.y:, dxy_to_crop_low.x:]
            self.foreground_mask = foreground_mask_cropped_low[:self.dxy_shape.y, :self.dxy_shape.x]
        else:
            self.foreground_mask = self.input_foreground_mask
        if must_crop and self.input_background_mask is not None:
            dxy_to_crop_low = self.xy_origin - self.xy_input_origin
            background_mask_cropped_low = self.input_background_mask[dxy_to_crop_low.y:, dxy_to_crop_low.x:]
            self.background_mask = background_mask_cropped_low[:self.dxy_shape.y, :self.dxy_shape.x]
        else:
            self.background_mask = self.input_background_mask

        # Invalidate aperture if entirely outside image:
        if self.dxy_shape.x == 0 or self.dxy_shape.y == 0:
            self.messages.append('Aperture lies outside image. Ap object invalid.')
            self.is_valid = False
            return

        # Invalidate aperture if any foreground pixels will lie outside image:
        if must_crop:
            active_pixels_as_input = np.sum(self.input_foreground_mask == False)
            active_pixels_cropped = np.sum(self.foreground_mask == False)
            if active_pixels_cropped != active_pixels_as_input:
                self.messages.append('Foreground pixels lie outside image. Ap object invalid.')
            self.is_valid = False
            return

        # Set background to None (zero value) if all active background pixels will lie outside image:
        if self.background_mask is not None:
            active_background_pixels = np.sum(self.background_mask == False)
            if active_background_pixels <= 0:
                self.background_mask = None

        # Compute pixel counts:
        self.foreground_pixel_count = np.sum(self.foreground_mask == False)  # active pixels
        if self.background_mask is None:
            self.background_level, self.background_std = 0.0, 0.0
            self.foreground_pixel_count = 0
            self.mask_overlap_pixel_count = 0
        else:
            self.background_level, self.background_std = calc_background_value(self.cutout,
                                                                               self.background_mask)
            self.background_pixel_count = np.sum(self.background_mask == False)  # active pixels
            self.mask_overlap_pixel_count = np.sum(np.logical_and((self.foreground_mask == False),
                                                                  (self.background_mask == False)))

        # Compute aperture statistics:
        self.raw_flux = np.sum(self.cutout, where=np.logical_not(self.foreground_mask))
        self.cutout_net = self.cutout - self.background_level
        self.net_flux = np.sum(self.cutout_net, where=np.logical_not(self.foreground_mask))
        self.stats = data_properties(data=self.cutout_net, mask=self.foreground_mask,
                                     background=self.background_level)
        self.is_cropped = must_crop
        self.is_valid = True

    def __str__(self):
        return 'Ap object at x,y = (' + str(self.xy_center.x) + ', ' + str(self.xy_center.y)

    @property
    def centroid(self):
        return self.stats.xcentroid, self.stats.ycentroid

    @property
    def flux_stddev(self, gain=1):
        """ Returns tuple of stats related to net flux of foreground pixels.
            Made a property so that gain can be passed in separately from Ap construction.
        :param gain: CCD-like gain in e-/ADU. Property of the specific camera (model).
                     Needed only for accurate uncertainty estimation. [float]
        :return: flux standard deviation. [float]
        """
        flux_variance_from_poisson_noise = self.raw_flux / gain  # from var(e-) = flux in e-.
        flux_variance_from_background = self.foreground_pixel_count * \
                                        ((self.background_std ** 2) / self.background_pixel_count)
        flux_variance = flux_variance_from_poisson_noise + flux_variance_from_background
        flux_stddev = sqrt(flux_variance)
        return flux_stddev

    @property
    def sigma(self):
        """ Returns width of flux, in pixels.
            Conservatively calculated as semi-major axis sigma of a best-fit ellipse to flux.
        :return: sigma width of flux, in pixels. [float]
        """
        return self.stats.semimajor_axis_sigma

    @property
    def fwhm(self):
        """ Returns full-width at half-max of flux, in pixels. Computed from .sigma().
        :return: estimate of FWHM full-width at half-maximum. [float]
        """
        return self.sigma * FWHM_PER_SIGMA

    def make_new_object(self, new_xy_center):
        """ Make new object from same image using new xy_center, with same mask shape.
            Used mostly by .recenter().
        :param new_xy_center: new x,y target center, in pixels relative to image origin. [XY object,
            or 2-tuple or list of length 2]
        """
        return Ap(self.image, new_xy_center, self.xy_origin, self.foreground_mask, self.background_mask)

    def recenter(self, max_adjustment=None, max_iterations=3):
        """ Subclasses should inherit this unchanged.
            (Whereas .make_new_object() must be written specially for each subclass).
        :param max_adjustment:
        :param max_iterations:
        :return: newly recentered object. [PointSourceAp object]
        """
        previous_ap, next_ap = self, self
        for i in range(max_iterations):
            previous_centroid = previous_ap.centroid
            new_xy_center = previous_centroid
            next_ap = self.make_new_object(new_xy_center)
            new_centroid = next_ap.centroid
            adjustment = sqrt((new_centroid[0] - previous_centroid[0])**2 +
                              (new_centroid[1] - previous_centroid[1])**2)
            if adjustment < max_adjustment:
                return next_ap
            previous_ap = next_ap
        return next_ap


class PointSourceAp(Ap):
    """ Standard photometric aperture for stationary point source of light, esp. for a star.
            Always makes a circular foreground mask and an annular background mask, both centered on the
            given image coordinates of the point source.
        (If we will need a background mask that bleeds to the cutout's edges rather than being restricted
            to an annulus--for example, to work close to the parent image's edges, then that will
            definitely require a new sibling class to this one so that recentering retains the mask shapes.)
    """
    def __init__(self, image, xy_center, foreground_radius, gap, background_width):
        """ Main and probably sole constructor for this class.
        :param image: the parent image array [numpy ndarray].
        :param xy_center: center pixel position in parent. This should be the best prior estimate
                   of the light source's centroid at mid-exposure, as (x,y) (not as numpy [y, x] array).
                   [XY object, 2-tuple, 2-list, or 2-array of floats]
        :param foreground_radius: radial size of foreground around point source, in pixels. [float]
        :param gap: width of gap, difference between radius of foreground and inside radius of
                        background annulus, in pixels. [float]
        :param background_width: width of annulus, difference between inside and outside radii
                   of background annulus, in pixels.
        """
        xy_center = xy_center if isinstance(xy_center, XY) else XY(xy_center[0], xy_center[1])
        self.foreground_radius = foreground_radius
        self.gap = gap
        self.background_width = background_width
        self.annulus_inner_radius = self.foreground_radius + self.gap
        self.annulus_outer_radius = self.annulus_inner_radius + self.background_width
        cutout_size = int(ceil(2 * self.annulus_outer_radius)) + 4  # generous, for safety.
        dxy_origin = DXY(int(round(xy_center.x - cutout_size / 2)),
                       int(round(xy_center.y - cutout_size / 2)))
        xy_center_in_cutout = xy_center - dxy_origin
        foreground_mask = make_circular_mask(mask_size=cutout_size, xy=tuple(xy_center_in_cutout),
                                             radius=self.foreground_radius)
        background_mask_center_disc = np.logical_not(make_circular_mask(cutout_size,
                                                                        tuple(xy_center_in_cutout),
                                                                        self.annulus_inner_radius))
        background_mask_outer_disc = make_circular_mask(cutout_size, tuple(xy_center_in_cutout),
                                                        self.annulus_outer_radius)
        background_mask = np.logical_or(background_mask_center_disc, background_mask_outer_disc)
        super().__init__(image, xy_center, dxy_origin, foreground_mask, background_mask)

    def make_new_object(self, new_xy_center):
        """ Make new object using new xy_center. Overrides parent-class method.
            Masks will be recreated by the constructor, using new xy_center.
         """
        return PointSourceAp(self.image, new_xy_center,
                             self.foreground_radius, self.gap, self.background_width)


class MovingSourceAp(Ap):
    """ Elongated 'pill-shaped' photometric aperture for moving point source of light,
            esp. for a minor planet/asteroid.
        (If we will need a background mask that bleeds to the cutout's edges rather than being restricted
            to a (pill-shaped) annulus--for example, to work close to the parent image's edges, that will
            definitely require a new sibling class to this one so that recentering retains the mask shapes.)
        """
    def __init__(self, image, xy_start, xy_end, foreground_radius, gap, background_width):
        """ Main and probably sole constructor for this class.
        :param image: the parent image array [numpy ndarray].
        :param xy_start: x,y pixel position in parent image of the beginning of the MP's motion.
               [XY object, 2-tuple, 2-list, or 2-array of floats]
        :param xy_end:x,y pixel position in parent image of the beginning of the MP's motion.
               [XY object, 2-tuple, 2-list, or 2-array of floats]
        :param foreground_radius: radius of the aperture source end-caps, in pixels.
               Does not include effect of MP motion. [float]
        :param gap: Gap in pixels between foreground mask and background mask. [float]
        :param background_width: Width in pixels of background mask. [float]
        """
        self.xy_start = xy_start if isinstance(xy_start, XY) else XY(xy_start[0], xy_start[1])
        self.xy_end = xy_end if isinstance(xy_end, XY) else XY(xy_end[0], xy_end[1])
        self.foreground_radius = foreground_radius
        self.gap = gap
        self.background_width = background_width
        self.background_inner_radius = self.foreground_radius + self.gap
        self.background_outer_radius = self.foreground_radius + self.gap + self.background_width
        xy_center = self.xy_start + (self.xy_end - self.xy_start) / 2
        corner_x_values = (xy_start.x + self.background_outer_radius,
                           xy_start.x - self.background_outer_radius,
                           xy_end.x + self.background_outer_radius,
                           xy_end.x - self.background_outer_radius)
        x_min = min(corner_x_values)
        x_max = max(corner_x_values)
        corner_y_values = (xy_start.y + self.background_outer_radius,
                           xy_start.y - self.background_outer_radius,
                           xy_end.y + self.background_outer_radius,
                           xy_end.y - self.background_outer_radius)
        y_min = min(corner_y_values)
        y_max = max(corner_y_values)
        cutout_size = DXY(int(round(x_max - x_min + 4)), int(round(y_max - y_min + 4)))
        dxy_offset = DXY(int(round(xy_center.x) - cutout_size.dx / 2.0),
                         int(round(xy_center.y) - cutout_size.dy / 2.0))
        xy_start_cutout = self.xy_start - dxy_offset
        xy_end_cutout = self.xy_end - dxy_offset
        foreground_mask = make_pill_mask(tuple(cutout_size),
                                         tuple(xy_start_cutout), tuple(xy_end_cutout),
                                         self.foreground_radius)
        background_inner_mask = make_pill_mask(tuple(cutout_size),
                                               tuple(xy_start_cutout), tuple(xy_end_cutout),
                                               self.background_inner_radius)
        background_outer_mask = make_pill_mask(tuple(cutout_size),
                                               tuple(xy_start_cutout), tuple(xy_end_cutout),
                                               self.background_outer_radius)
        background_mask = np.logical_or(background_outer_mask,
                                        np.logical_not(background_inner_mask))
        super().__init__(image, xy_center, dxy_offset, foreground_mask, background_mask)


_____IMAGE_and_GEOMETRY_SUPPORT____________________________________ = 0


def calc_ra_limits(ra_list):
    """ For list of RA values (usually RAs of image corners), return min and max RA within [0, 360). """
    # TODO: this algorithm is O(2) (ouch). Try sorting RAs first, then looking left one element in the list.
    min_ra, max_diff = 0.0, 360.0
    for this_ra in ra_list:
        other_ras = [ra for ra in ra_list.copy() if ra != this_ra]
        this_max_diff = max([(ra - this_ra) % 360 for ra in other_ras])
        if this_max_diff < max_diff:
            min_ra, max_diff = this_ra, this_max_diff
    ra_deg_min = min_ra
    ra_deg_max = (ra_deg_min + max_diff) % 360
    return ra_deg_min, ra_deg_max


def aggregate_bounding_ra_dec(fits_objects, extension_percent=3):
    """ For list of plate-solved FITS-class objects, return aggregate bounding RA and Dec values.
        Typically needed for catalog lookup, to cover the area of a group of nearly colocated images.
    :param fits_objects: FITS-class object for images of interest. [iterable of FITS-class objects]
    :param extension_percent: how much to extend bounding box, as percent of each FITS image size
           (not as percent of overall aggregate bounding box size). [float]
    :return: tuple of results: ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max. [tuple of 4 floats]
    """
    ra_list, dec_list = [], []
    for fo in fits_objects:
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = \
            fo.bounding_ra_dec(extension_percent=extension_percent)
        ra_list.extend([ra_deg_min, ra_deg_max])
        dec_list.extend([dec_deg_min, dec_deg_max])
    aggr_ra_deg_min, aggr_ra_deg_max = calc_ra_limits(ra_list)
    aggr_dec_deg_min, aggr_dec_deg_max = min(dec_list), max(dec_list)
    return aggr_ra_deg_min, aggr_ra_deg_max, aggr_dec_deg_min, aggr_dec_deg_max


def calc_cutout_offsets(xy_center, cutout_radius):
    """ Calculate and return the index x- and y-offsets from a parent image to a cutout of given radius.
        This is made a separate function so that all cutout-dependent classes must use *identical* offsets.
    """
    x_offset = int(round(xy_center[0]) - cutout_radius)
    y_offset = int(round(xy_center[1]) - cutout_radius)
    return x_offset, y_offset


def make_circular_mask(mask_size, xy, radius):
    """ Construct a traditional mask array for small, stationary object, esp. for a star.
        Unmask only those pixels *within* radius pixels of a given point. Invert the mask separately to
            mask the interior. Convention: pixel True -> VALID (opposite of numpy).
    :param mask_size: edge size of new mask array. [int]
    :param xy: (x, y) pixel coordinates of central point, relative to mask origin. [2-tuple of floats]
    :param radius: radius of ends and half-width of center region. [float]
    :return: mask array, True -> VALID (opposite convention from numpy). [np.ndarray of booleans]
    """
    circle = Circle_in_2D(xy_origin=xy, radius=radius)
    new_mask = circle.contains_points_unitgrid(0, mask_size -1, 0, mask_size - 1, include_edges=True)
    return new_mask


def make_pill_mask(mask_size, xya, xyb, radius):
    """ Construct a mask array for MP in motion: unmask only those pixels within radius pixels of
        any point in line segment from xya to xyb. Convention: pixel True -> VALID (opposite of numpy).
    :param mask_size: (x,y) size of array to generate. [2-tuple of ints]
    :param xya: (xa, ya) pixel coordinates of start-motion point. [XY object, or 2-tuple of floats]
    :param xyb: (xb, yb) pixel coordinates of end-motion point. [XY object, or 2-tuple of floats]
    :param radius: radius of ends and half-width of center region. [float]
    :return: mask array, True -> VALID (opposite convention from numpy). [np.ndarray of booleans]
    """
    xya = xya if isinstance(xya, XY) else XY(xya[0], xya[1])
    xyb = xyb if isinstance(xyb, XY) else XY(xyb[0], xyb[1])
    if xya == xyb:
        return make_circular_mask(max(mask_size), xya, radius)

    # Make circle and rectangle objects:
    circle_a = Circle_in_2D(xya, radius)
    circle_b = Circle_in_2D(xyb, radius)
    dxy_ab = xyb - xya
    length_ab = dxy_ab.length
    dxy_a_corner1 = (radius / length_ab) * DXY(dxy_ab.dy, -dxy_ab.dx)  # perpendicular to ab vector.
    dxy_a_corner2 = (radius / length_ab) * DXY(-dxy_ab.dy, dxy_ab.dx)  # "
    xy_corner1 = xya + dxy_a_corner1
    xy_corner2 = xya + dxy_a_corner2
    xy_corner3 = xyb + dxy_a_corner2
    rectangle = Rectangle_in_2D(xy_corner1, xy_corner2, xy_corner3)

    # Make mask, including edges so no gaps can appear at rectangle corners:
    circle_mask_a = circle_a.contains_points_unitgrid(0, mask_size[0] - 1, 0, mask_size[1] - 1, True)
    circle_mask_b = circle_b.contains_points_unitgrid(0, mask_size[0] - 1, 0, mask_size[1] - 1, True)
    rectangle_mask = rectangle.contains_points_unitgrid(0, mask_size[0] - 1, 0, mask_size[1] - 1, True)
    mask = np.logical_and(np.logical_and(circle_mask_a, circle_mask_b), rectangle_mask)
    return mask


def distance_to_line(xy_pt, xy_a, xy_b, dist_ab=None):
    """ Yield the closest (perpendicular) distance from point (xpt, ypt) to the line (not necessarily
        within the closed line segment) passing through (x1,y1) and (x2,y2). """
    # TODO: consider moving to geometry module.
    xpt, ypt = tuple(xy_pt)
    xa, ya = tuple(xy_a)
    xb, yb = tuple(xy_b)
    if dist_ab is None:
        dist_ab = sqrt((yb - ya)**2 + (xb - xa)**2)
    distance = abs((yb - ya) * xpt - (xb - xa) * ypt + xb * ya - yb * xa) / dist_ab
    return distance


def calc_background_value(data, mask=None, dilate_size=3):
    """ Calculate the sigma-clipped median value of a (possibly masked) data array.
    :param data: array of pixels. [2-D ndarray of floats]
    :param mask: mask array, True=masked, i.e., use only False pixels. [None, or 2-D nadarray of bool]
    :param dilate_size: the number of pixels out from a detected outlier pixels to also mask.
        If sample is heavily undersampled (e.g., FWHM < 3 pixels), one might try 1-2 pixels.
        If image is heavily oversampled (e.g., FWHM > 10-15 pixels), one might increase this
        to perhaps 0.25-0.5 FWHM. [int]
    :return: tuple of background adu level (flux per pixel), standard deviation within used pixels.
                 [2-tuple of floats]
    """
    if mask is None:  # use all pixels.
        this_mask = np.full_like(data, False, dtype=np.bool)
    elif mask.shape != data.shape:  # bad mask shape.
        return None
    elif np.sum(mask == False) == 0:  # no valid pixels.
        return 0.0, 0.0
    else:
        this_mask = mask.copy()
    # this_mask: user-supplied mask of pixels simply not to consider at all.
    # source_mask: masks out detected light-source pixels. MAY INCLUDE pixels masked out by this_mask!
    # stats_mask: the intersection of valid pixels from this_mask and source_mask.
    # npixels = 2, which will mask out practically all cosmic ray hits.
    # filter_fwhm=2, small enough to still capture cosmic ray hits, etc.
    #     (make_source_mask() will probably fail if fwhm < 2.)
    # dilate_size=3, small enough to preserve most background pixels, but user may override.
    source_mask = make_source_mask(data, mask=this_mask, nsigma=2, npixels=5,
                                   filter_fwhm=2, dilate_size=int(dilate_size))
    stats_mask = np.logical_or(this_mask, source_mask)
    _, median, std = sigma_clipped_stats(data, sigma=3.0, mask=stats_mask)
    return median, std


_____FITS_FILE_HANDLING_______________________________________________ = 0


def all_fits_filenames(top_directory, rel_directory, validate_fits=False):
    """  Return list of all FITS filenames (name.extension; no directory info) in given directory_path.
         (Code for this exists already, somewhere.)
    :param top_directory:
    :param rel_directory:
    :param validate_fits: If True, open FITS files and include only if valid.
        If False, include filename if it appears valid without opening the FITS file.
    :return: List of all FITS filenames in given directory_path [list of strings]
    """
    # TODO: write all_fits_files().
    pass

