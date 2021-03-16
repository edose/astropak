__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import numpy as np
from datetime import datetime, timezone, timedelta
import pytest
import astropy.io.fits as apyfits

# Other astropak modules:
from astropak import image  # TARGET TEST MODULE.
from astropak.util import RaDec, dec_as_degrees, ra_as_degrees
from astropak.geometry import XY, DXY

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")


class Test_Class_FITS:

    def test_constructor(self):
        test_rel_directory = '$data_for_test'

        # Test failure on missing file:
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename='$no_file.txt')
        assert fits.is_valid is False

        # Test exception handling for non-FITS file format:
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename='dummy.txt')
        assert fits.is_valid is False

        # Open FITS file with no calibration (at least not by MaxIm 5/6) and no plate solution:
        given_filename = 'AD Dra-S001-R001-C001-I.fts'
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename=given_filename)
        assert fits.is_valid
        assert fits.is_calibrated is False
        assert fits.is_plate_solved is False
        assert fits.plate_solution_is_pinpoint is False

        # Test constructor from path in pieces, known extension:
        given_filename = 'CE Aur-0001-V.fts'
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename=given_filename)
        assert fits.is_valid
        # Verify fields:
        assert fits.fullpath == os.path.join(TEST_TOP_DIRECTORY, test_rel_directory, given_filename)
        assert fits.object == 'CE Aur'
        assert fits.is_calibrated
        assert fits.is_plate_solved
        assert fits.plate_solution_is_pinpoint
        assert fits.focal_length == pytest.approx(2702, abs=1)
        assert fits.exposure == pytest.approx(587, abs=1)
        assert fits.temperature == pytest.approx(-35, abs=0.1)
        target_start_utc = datetime(2017, 4, 24, 4, 0, 31).replace(tzinfo=timezone.utc)
        diff_seconds = (fits.utc_start - target_start_utc).total_seconds()
        assert abs(diff_seconds) < 1
        target_mid_utc = target_start_utc + timedelta(seconds=fits.exposure / 2.0)
        diff_seconds = (fits.utc_mid - target_mid_utc).total_seconds()
        assert abs(diff_seconds) < 1
        assert fits.filter == 'V'
        assert fits.airmass == pytest.approx(1.5263, abs=0.0001)
        assert fits.guide_exposure == pytest.approx(1.4, abs=0.001)
        assert fits.fwhm == pytest.approx(5.01, abs=0.01)
        assert fits.plate_solution['CD1_1'] == pytest.approx(-1.92303985969E-006)
        assert fits.plate_solution['CD2_1'] == pytest.approx(1.90588522664E-004)
        assert fits.plate_solution['CRVAL1'] == pytest.approx(1.03834010522E+002)
        assert fits.ra == pytest.approx(103.83791666666667)
        assert fits.dec == pytest.approx(46.28638888888889)
        # Verify field .image (== .image_xy):
        assert fits.image.shape == (3072, 2047)  # *image* (x,y), which is *array* (n_rows, n_columns)
        assert fits.image[0, 0] == 275  # upper-left corner
        assert fits.image[0, 2046] == 180  # lower-left corner
        assert fits.image[3071, 2046] == 265  # lower-right corner
        assert fits.image[3071, 0] == 285  # upper-right corner
        # Test methods:
        assert fits.header_has_key('NAXIS')
        assert not fits.header_has_key('NOT A KEY')
        assert fits.header_value('NAXIS1') == 3072  # int
        assert fits.header_value('EXPOSURE') == pytest.approx(587.0)  # float
        assert fits.header_value('OBJECT').strip() == 'CE Aur'
        radec = RaDec('06:56:12.8', '+46:32:08.9')
        x, y = fits.xy_from_radec(radec)
        assert x == pytest.approx(2826, 0.5)  # according to WCS terms only.
        assert y == pytest.approx(218, 0.5)   # "
        radec = fits.radec_from_xy(846, 932)
        assert radec.ra == pytest.approx(ra_as_degrees('06:55:26.58'), 0.001)
        assert radec.dec == pytest.approx(dec_as_degrees('46:09:25.6'), 0.001)
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = fits.bounding_ra_dec(extension_percent=2)
        assert ra_deg_min == pytest.approx(103.5359646, abs=0.001)
        assert ra_deg_max == pytest.approx(104.1320565, abs=0.001)
        assert dec_deg_min == pytest.approx(45.9816651, abs=0.001)
        assert dec_deg_max == pytest.approx(46.5946668, abs=0.001)

        # Test constructor when giving only 2 path items:
        given_filename = 'CE Aur-0001-V.fts'
        this_directory = os.path.join(TEST_TOP_DIRECTORY, test_rel_directory)
        fits = image.FITS(this_directory, given_filename)
        assert fits.is_valid
        assert fits.fullpath == os.path.join(this_directory, given_filename)
        assert fits.ra == pytest.approx(103.83791666666667)
        assert fits.dec == pytest.approx(46.28638888888889)

        # Test constructor when giving only fullpath:
        given_filename = 'CE Aur-0001-V.fts'
        fullpath = os.path.join(TEST_TOP_DIRECTORY, test_rel_directory, given_filename)
        fits = image.FITS(fullpath)
        assert fits.is_valid
        assert fits.fullpath == os.path.join(this_directory, given_filename)
        assert fits.ra == pytest.approx(103.83791666666667)
        assert fits.dec == pytest.approx(46.28638888888889)

        # Open FITS file without known FITS extension (which FITS constructor itself must determine):
        given_filename = 'CE Aur-0001-V'
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename=given_filename)
        assert fits.is_valid
        assert fits.fullpath == os.path.join(TEST_TOP_DIRECTORY, test_rel_directory, given_filename +'.fts')
        assert fits.object == 'CE Aur'
        assert fits.airmass == pytest.approx(1.5263, abs=0.0001)

    def test_parm_pinpoint_pixel_scale_multiplier(self):
        test_rel_directory = '$data_for_test'
        # Test parm pinpoint_pixel_scale_multiplier:
        pinpoint_pixel_scale_multiplier = 0.68198 / 0.68618  # = pxscale(WCS) / pxscale(PinPoint).
        given_filename = 'CE Aur-0001-V.fts'
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename=given_filename,
                          pinpoint_pixel_scale_multiplier=pinpoint_pixel_scale_multiplier)
        assert fits.is_valid
        assert fits.plate_solution_is_pinpoint
        assert fits.plate_solution['CD1_1'] == pytest.approx(-1.92303985969E-006 *
                                                             pinpoint_pixel_scale_multiplier)
        assert fits.plate_solution['CD1_1'] != pytest.approx(-1.92303985969E-006)
        assert fits.plate_solution['CD2_1'] == pytest.approx(1.90588522664E-004 *
                                                             pinpoint_pixel_scale_multiplier)
        assert fits.plate_solution['CRVAL1'] == pytest.approx(1.03834010522E+002)  # unchanged
        radec = RaDec('06:56:12.8', '+46:32:08.9')
        x, y = fits.xy_from_radec(radec)
        assert x == pytest.approx(2834, 0.5)  # according to WCS terms only.
        assert y == pytest.approx(213, 0.5)   # "
        radec = fits.radec_from_xy(846, 932)
        assert radec.ra == pytest.approx(ra_as_degrees('06:55:26.58'), abs=0.001)
        assert radec.dec == pytest.approx(dec_as_degrees('46:09:25.6'), abs=0.001)

    def test_fits__xy_from_radec(self):
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test',
                          filename='CE Aur-0001-V.fts')
        # All tests lack distortion corrections (as none available in FITS header),
        #    and so in real images calculated (x,y) values at edges will not quite line up with stars.
        radec_near_center = RaDec('06:55:21.25', '+46:17:33.0')
        x, y = fits.xy_from_radec(radec_near_center)
        assert list((x, y)) == pytest.approx([1557.6, 1005.8], abs=0.25)

        radec_upper_left = RaDec('06:56:10.6', '+46:02:27.1')
        x, y = fits.xy_from_radec(radec_upper_left)
        assert list((x, y)) == pytest.approx([229.8, 270.3], abs=0.25)

        radec_upper_right = RaDec('06:56:14.3', '+46:29:11.6')
        x, y = fits.xy_from_radec(radec_upper_right)
        assert list((x, y)) == pytest.approx([2567.6, 197.2], abs=0.25)

        radec_lower_left = RaDec('06:54:26.0', '+46:02:44.9')
        x, y = fits.xy_from_radec(radec_lower_left)
        assert list((x, y)) == pytest.approx([271.8, 1857.0], abs=0.25)

        radec_lower_right = RaDec('06:54:18.0', '+46:30:02.0')
        x, y = fits.xy_from_radec(radec_lower_right)
        assert list((x, y)) == pytest.approx([2658.7, 1946.6], abs=0.25)

    def test_fits_bounding_ra_dec(self):
        # Normal case:
        fits = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        assert fits.bounding_ra_dec(extension_percent=0) == pytest.approx([103.5474, 104.1206,
                                                                           45.9935, 46.5829], abs=0.0001)
        assert fits.bounding_ra_dec(extension_percent=3) == pytest.approx([103.5302, 104.1378,
                                                                           45.9758, 46.6006], abs=0.0001)
        # Case: Image straddles RA of zero:
        fits.plate_solution['CRVAL1'] = 0.1
        assert fits.bounding_ra_dec(extension_percent=0) == pytest.approx([359.8134, 0.3866,
                                                                           45.9935, 46.5829], abs=0.0001)


class Test_Class_Ap:
    """ This will test items of class Ap that cannot be tested via most subclasses."""
    @staticmethod
    @pytest.fixture(scope='function')
    def get_image():
        """ Return test header and image, only. """
        fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
        hdu0 = apyfits.open(fullpath)[0]
        hdr = hdu0.header
        im = hdu0.data
        return hdr, im

    def test_constructor_mask_overlap(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        background_mask = foreground_mask.copy()
        foreground_mask[25:28, 11:17] = False
        background_mask[26:30, 9:14] = False
        ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                      foreground_mask=foreground_mask, background_mask=background_mask)
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is True
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 6

    def test_constructor_background_mask_none(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        foreground_mask[25:28, 11:17] = False
        ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                      foreground_mask=foreground_mask, background_mask=None)
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is True
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Verify background mask:
        foreground_mask_inverse = np.logical_not(foreground_mask.copy())
        assert np.array_equal(ap.background_mask, foreground_mask_inverse)

    def test_constructor_background_mask_zero(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        foreground_mask[25:28, 11:17] = False
        ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                      foreground_mask=foreground_mask, background_mask=0)
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is True
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Verify background mask:
        expected_background_mask = np.full(shape=(41, 41), fill_value=True)
        assert np.array_equal(ap.background_mask, expected_background_mask)

    def test_ids(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        background_mask = foreground_mask.copy()
        foreground_mask[25:28, 11:17] = False
        background_mask[26:30, 9:14] = False
        ap1 = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                       foreground_mask=foreground_mask, background_mask=background_mask,
                       source_id='comp star 1', obs_id='345')
        ap2 = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                       foreground_mask=foreground_mask, background_mask=background_mask,
                       source_id='MP_3456', obs_id='haha')
        ap3 = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                       foreground_mask=foreground_mask, background_mask=background_mask)
        assert (ap1.source_id, ap1.obs_id) == ('comp star 1', '345')
        assert (ap2.source_id, ap2.obs_id) == ('MP_3456', 'haha')
        assert (ap3.source_id, ap3.obs_id) == ('', '')


    def test_constructor_exceptions(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        background_mask = np.full(shape=(40, 41), fill_value=True)
        with pytest.raises(image.MaskError) as e:
            ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                          foreground_mask=foreground_mask, background_mask=background_mask)
        assert 'masks differ in shape' in str(e)
        with pytest.raises(image.MaskError) as e:
            ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                          foreground_mask=foreground_mask, background_mask=True)
        assert 'mask type' in str(e) and 'bool' in str(e) and 'not valid' in str(e)

    def test_make_new_object(self, get_image):
        hdr, im = get_image
        foreground_mask = np.full(shape=(41, 41), fill_value=True)
        background_mask = foreground_mask.copy()
        foreground_mask[25:28, 11:17] = False
        background_mask[26:30, 9:14] = False
        ap = image.Ap(im, xy_center=(1000, 1100), xy_offset=(980, 1080),
                      foreground_mask=foreground_mask, background_mask=background_mask)
        with pytest.raises(NotImplementedError):
            _ = ap.make_new_object(new_xy_center=(445, 667))


class Test_Class_PointSourceAp:
    """ This will test class PointSourceAp, but will also test the parent class Ap pretty thoroughly."""
    @staticmethod
    @pytest.fixture(scope='function')
    def get_image():
        """ Return test header and image, only. """
        fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
        hdu0 = apyfits.open(fullpath)[0]
        hdr = hdu0.header
        im = hdu0.data
        return hdr, im

    def test_constructor_masks_inside_image(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(1476.3, 1243.7),
                                 foreground_radius=9, gap=6, background_width=5)
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is True
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Values as input:
        assert (ap.foreground_radius, ap.gap, ap.background_width) == (9, 6, 5)
        assert ap.annulus_inner_radius == ap.foreground_radius + ap.gap
        assert ap.annulus_outer_radius == ap.foreground_radius + ap.gap + ap.background_width
        assert np.array_equal(ap.image, im)
        assert ap.xy_center == XY(1476.3, 1243.7)
        assert ap.input_foreground_mask.shape == (44, 44)
        assert ap.input_background_mask.shape == ap.input_foreground_mask.shape
        # Shapes and pixel counts:
        assert ap.cutout.shape == ap.foreground_mask.shape == ap.background_mask.shape == (44, 44)
        assert ap.foreground_pixel_count == np.sum(ap.input_foreground_mask == False) == 255
        assert ap.background_pixel_count == np.sum(ap.input_background_mask == False) == 549
        # ADUs and fluxes:
        assert ap.background_level == pytest.approx(257, abs=1)
        assert ap.background_std == pytest.approx(13.84, abs=0.1)
        assert ap.foreground_max == 1065
        assert ap.foreground_min == 234
        assert ap.raw_flux == pytest.approx(96364, abs=10)
        assert ap.net_flux == pytest.approx(30829, abs=10)
        assert ap.flux_stddev(gain=1.57) == pytest.approx(247.95, abs=0.1)
        # Source flux position & shape:
        assert ap.xy_centroid[0] == pytest.approx(1476.23, abs=0.01)
        assert ap.xy_centroid[1] == pytest.approx(1243.39, abs=0.01)
        assert ap.sigma == pytest.approx(2.81, abs=0.1)
        assert ap.fwhm == pytest.approx(6.63, abs=0.1)
        assert ap.elongation == pytest.approx(1.086, abs=0.02)

    def test_constructor_background_cropped_low(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(17, 15),
                                 foreground_radius=9, gap=6, background_width=5)  # no source there.
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is False
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Shapes and pixel counts:
        assert ap.cutout.shape == ap.foreground_mask.shape == ap.background_mask.shape == (37, 39)
        assert ap.foreground_pixel_count == np.sum(ap.foreground_mask == False) == 253
        assert ap.background_pixel_count == np.sum(ap.background_mask == False) == 440
        # ADUs and fluxes:
        assert ap.background_level == pytest.approx(246, abs=1)
        assert ap.background_std == pytest.approx(23, abs=1)
        assert ap.foreground_max == 312
        assert ap.foreground_min == 186
        assert ap.raw_flux == pytest.approx(62611, abs=10)

    def test_constructor_background_cropped_high(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(3060, 2035),
                                 foreground_radius=9, gap=6, background_width=5)  # no source there.
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is False
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Shapes and pixel counts:
        assert ap.cutout.shape == ap.foreground_mask.shape == ap.background_mask.shape == (34, 34)
        assert ap.foreground_pixel_count == np.sum(ap.foreground_mask == False) == 253
        assert ap.background_pixel_count == np.sum(ap.background_mask == False) == 261
        # ADUs and fluxes:
        assert ap.background_level == pytest.approx(245, abs=1)
        assert ap.background_std == pytest.approx(18, abs=1)
        assert ap.foreground_max == 300
        assert ap.foreground_min == 193
        assert ap.raw_flux == pytest.approx(61791, abs=10)

    def test_constructor_foreground_cropped(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(2568, 2050),
                                 foreground_radius=9, gap=6, background_width=5)  # no source there.
        # Quality criteria:
        assert ap.is_valid is False
        assert ap.all_inside_image is False
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is True
        assert ap.mask_overlap_pixel_count == 0

    def test_constructor_ap_all_outside(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(2568, 2100),
                                 foreground_radius=9, gap=6, background_width=5)
        # Quality criteria:
        assert ap.is_valid is False
        assert ap.all_inside_image is False
        assert ap.all_outside_image is True
        assert ap.any_foreground_outside_image is True
        assert ap.mask_overlap_pixel_count is None

    def test_constructor_ids(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(1476.3, 1243.7),
                                 foreground_radius=9, gap=6, background_width=5,
                                 source_id='xx', obs_id='xyz')
        assert (ap.source_id, ap.obs_id) == ('xx', 'xyz')

    def test_make_new_object(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(1476, 1244),
                                 foreground_radius=9, gap=6, background_width=5)
        ap2 = ap.make_new_object(new_xy_center=(2086, 1570))
        assert ap2.xy_center == (2086, 1570)
        assert np.array_equal(ap.foreground_mask, ap2.foreground_mask)
        assert np.array_equal(ap.background_mask, ap2.background_mask)
        assert ap2.xy_centroid[0] == pytest.approx(2085.70, abs=0.02)
        assert ap2.xy_centroid[1] == pytest.approx(1569.12, abs=0.02)
        assert ap2.foreground_max == 3414

    def test_recenter(self, get_image):
        """ This will test class Ap.recenter() for all subclasses. """
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(1476.3, 1243.7),
                                 foreground_radius=9, gap=6, background_width=5)
        ap2 = ap.recenter()
        assert ap2.is_valid
        assert ap2.xy_center == ap.xy_centroid
        assert ap2.xy_centroid == pytest.approx((1476.310, 1243.271), abs=0.002)


class Test_Class_MovingSourceAp:
    """ This will test class MovingSourceAp.
        Assumes class Ap was tested thoroughly by Test_Class_PointSourceAp."""
    @staticmethod
    @pytest.fixture(scope='function')
    def get_image():
        """ Return test header and image, only. """
        fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
        hdu0 = apyfits.open(fullpath)[0]
        hdr = hdu0.header
        im = hdu0.data
        return hdr, im

    def test_constructor_masks_inside_image(self, get_image):
        hdr, im = get_image
        ap = image.MovingSourceAp(im, xy_start=(2354, 1505), xy_end=(2361, 1510),
                                  foreground_radius=9, gap=6, background_width=5)
        # Quality criteria:
        assert ap.is_valid is True
        assert ap.all_inside_image is True
        assert ap.all_outside_image is False
        assert ap.any_foreground_outside_image is False
        assert ap.mask_overlap_pixel_count == 0
        # Values as input:
        assert (ap.foreground_radius, ap.gap, ap.background_width) == (9, 6, 5)
        assert np.array_equal(ap.image, im)
        assert ap.xy_start == XY(2354, 1505)
        assert ap.xy_end == XY(2361, 1510)
        # Shapes and pixel counts:
        assert ap.cutout.shape == ap.foreground_mask.shape == ap.background_mask.shape == (49, 51)  # (y,x)
        assert ap.foreground_pixel_count == 408
        assert ap.foreground_pixel_count == np.sum(ap.input_foreground_mask == False)
        assert ap.background_pixel_count == 633
        assert ap.background_pixel_count == np.sum(ap.input_background_mask == False)
        # ADUs and fluxes:
        assert ap.background_level == pytest.approx(254, abs=1)
        assert ap.background_std == pytest.approx(15.1, abs=0.2)
        assert ap.foreground_max == 1817
        assert ap.foreground_min == 224
        assert ap.raw_flux == pytest.approx(210000, abs=100)
        assert ap.net_flux == pytest.approx(106368, abs=100)
        assert ap.flux_stddev(gain=1.57) == pytest.approx(366, abs=1)
        # Source flux position & shape:
        assert ap.xy_centroid[0] == pytest.approx(2357.56, abs=0.05)
        assert ap.xy_centroid[1] == pytest.approx(1507.14, abs=0.05)
        assert ap.sigma == pytest.approx(3.14, abs=0.05)
        assert ap.fwhm == pytest.approx(7.40, abs=0.05)

    def test_constructor_ids(self, get_image):
        hdr, im = get_image
        ap = image.PointSourceAp(im, xy_center=(1476.3, 1243.7),
                                 foreground_radius=9, gap=6, background_width=5,
                                 source_id='haha', obs_id='hoho')
        assert (ap.source_id, ap.obs_id) == ('haha', 'hoho')

    def test_make_new_object(self, get_image):
        hdr, im = get_image
        ap = image.MovingSourceAp(im, xy_start=(2354, 1505), xy_end=(2360, 1511),
                                  foreground_radius=9, gap=6, background_width=5,
                                  source_id='1', obs_id='22')
        ap2 = ap.make_new_object(new_xy_center=(2086, 1570))
        assert ap2.xy_center == (2086, 1570)
        assert np.array_equal(ap.foreground_mask, ap2.foreground_mask)
        assert np.array_equal(ap.background_mask, ap2.background_mask)
        assert ap2.xy_centroid[0] == pytest.approx(2085.66, abs=0.02)
        assert ap2.xy_centroid[1] == pytest.approx(1569.07, abs=0.02)
        assert ap2.foreground_max == 3414


_____IMAGE_and_GEOMETRY_SUPPORT____________________________________ = 0


def test_calc_ra_limits():
    """ Verify that fn gives same result regardless of RA list order. """
    ra_list = [23.3, 23.7, 23.4, 23.1]  # normal case.
    for _ in ra_list:
        ra_list.append(ra_list.pop(0))
        assert list(image.calc_ra_limits(ra_list)) == pytest.approx([23.1, 23.7])
    ra_list = [0.3, 359.7, 359.8, 0.2]  # RA straddling zero.
    for _ in ra_list:
        ra_list.append(ra_list.pop(0))
        assert list(image.calc_ra_limits(ra_list)) == pytest.approx([359.7, 0.3])


def test_aggregate_bounding_ra_dec():
    # Normal case:
    for extension_percent in (0, 3):
        fits1 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits2 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits2.plate_solution['CRVAL1'] += 0.11  # RA
        fits2.plate_solution['CRVAL2'] -= 0.13  # Dec
        fits3 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits3.plate_solution['CRVAL1'] -= 0.14
        fits3.plate_solution['CRVAL2'] += 0.09
        fits_objects = [fits1, fits2, fits3]
        aggr_ra_deg_min, aggr_ra_deg_max, aggr_dec_deg_min, aggr_dec_deg_max = \
            image.aggregate_bounding_ra_dec(fits_objects, extension_percent=extension_percent)
        expected_aggr_ra_deg_min = fits3.bounding_ra_dec(extension_percent)[0]
        expected_aggr_ra_deg_max = fits2.bounding_ra_dec(extension_percent)[1]
        expected_aggr_dec_deg_min = fits2.bounding_ra_dec(extension_percent)[2]
        expected_aggr_dec_deg_max = fits3.bounding_ra_dec(extension_percent)[3]
        assert aggr_ra_deg_min == pytest.approx(expected_aggr_ra_deg_min, abs=0.0001)
        assert aggr_ra_deg_max == pytest.approx(expected_aggr_ra_deg_max, abs=0.0001)
        assert aggr_dec_deg_min == pytest.approx(expected_aggr_dec_deg_min, abs=0.0001)
        assert aggr_dec_deg_max == pytest.approx(expected_aggr_dec_deg_max, abs=0.0001)

    # Case: fits images straddle RA zero:
    for extension_percent in (0, 3):
        fits1 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits1.plate_solution['CRVAL1'] = 0.1  # RA
        fits2 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits2.plate_solution['CRVAL1'] = 0.13  # RA
        fits2.plate_solution['CRVAL2'] += 0.04  # Dec
        fits3 = image.FITS(TEST_TOP_DIRECTORY, rel_directory='$data_for_test', filename='CE Aur-0001-V.fts')
        fits3.plate_solution['CRVAL1'] = 359.96  # RA
        fits3.plate_solution['CRVAL2'] -= 0.053  # Dec
        fits_objects = [fits1, fits2, fits3]
        aggr_ra_deg_min, aggr_ra_deg_max, aggr_dec_deg_min, aggr_dec_deg_max = \
            image.aggregate_bounding_ra_dec(fits_objects, extension_percent=extension_percent)
        expected_aggr_ra_deg_min = fits3.bounding_ra_dec(extension_percent)[0]
        expected_aggr_ra_deg_max = fits2.bounding_ra_dec(extension_percent)[1]
        expected_aggr_dec_deg_min = fits3.bounding_ra_dec(extension_percent)[2]
        expected_aggr_dec_deg_max = fits2.bounding_ra_dec(extension_percent)[3]
        assert aggr_ra_deg_min == pytest.approx(expected_aggr_ra_deg_min, abs=0.0001)
        assert aggr_ra_deg_max == pytest.approx(expected_aggr_ra_deg_max, abs=0.0001)
        assert aggr_dec_deg_min == pytest.approx(expected_aggr_dec_deg_min, abs=0.0001)
        assert aggr_dec_deg_max == pytest.approx(expected_aggr_dec_deg_max, abs=0.0001)


def test_distance_to_line():  # OK 2021-01-28.
    assert image.distance_to_line((3, 4), (-1, 7), (-1, -33)) == 4.0  # horizontal line.
    assert image.distance_to_line((35, 4), (-100, 7), (100, 7)) == 3.0  # verical line.
    assert image.distance_to_line((15, 12), (-12, 43), (13, -17)) == 13.0  # normal within line segment.
    assert image.distance_to_line((15, 12), (-12, 43), (23, -41)) == 13.0  # normal outside line segment.
    assert image.distance_to_line((3, 4), (-1, 7), (-1, -33), dist_ab=40.0) == 4.0


def test_make_circular_mask():
    cm = image.make_circular_mask(mask_size=25, xy_origin=(12.2, 15.5), radius=8)
    assert isinstance(cm, np.ndarray)
    assert cm.dtype == np.bool
    assert cm.shape == (25, 25)
    assert np.sum(cm == False) == 200  # number of valid pixels.
    # assert not np.any(cm[12:18, 8:14])
    assert not np.any(cm[8:14, 12:18])  # numpy index and mask-boolean conventions.


def test_make_pill_mask():
    pm = image.make_pill_mask(mask_shape_xy=(41, 41), xya=(7, 20), xyb=(11, 15), radius=4)
    assert isinstance(pm, np.ndarray)
    assert pm.shape == (41, 41)
    assert np.sum(pm == False) == 100  # number of valid pixels.

    pm = image.make_pill_mask((43, 44), (10, 20), (10, 18), 7)  # vertical motion.
    assert pm.shape == (44, 43)
    assert np.sum(pm == False) == 179  # number of valid pixels.

    pm = image.make_pill_mask((61, 60), (13, 20), (28, 20), 9)  # horizontal motion, increasing x.
    assert pm.shape == (60, 61)
    assert np.sum(pm == False) == 538  # number of valid pixels.

    pm_reverse = image.make_pill_mask((61, 60), (28, 20), (13, 20), 9)  # horizontal motion, decreasing x.
    assert np.array_equal(pm_reverse, pm)  # doesn't matter which point is the start, which is end.


def test_calc_background_value():
    # Test simple case, no excluded pixels, no mask:
    im, _ = np.meshgrid(np.arange(10), np.arange(12))
    im = im.copy()
    median, std = image.calc_background_value(im)
    assert median == 4.5
    assert std == pytest.approx(2.872, abs=0.005)

    # Test simple case, no mask:
    im[5, 3:7] = 40  # outlier pixels.
    median, std = image.calc_background_value(im)
    assert median == 4.0
    assert std == pytest.approx(3.136, abs=0.005)

    # Test with added mask:
    mask = np.array(im <= 1.0)
    median, std = image.calc_background_value(im, mask)
    assert median == 6.0
    assert std == pytest.approx(2.445, abs=0.005)

    # Test uniform pixel (problematic) case:
    im = np.full(shape=(10, 20), fill_value=7, dtype=np.float).copy()
    median, std = image.calc_background_value(im)
    assert median == 7.0
    assert std == 0.0

    # Ensure dilate_size default, and that it gets used as an integer:
    assert image.calc_background_value(im) == \
           image.calc_background_value(im, dilate_size=3) == \
           image.calc_background_value(im, dilate_size=3.3)



_____Test_AP_CLASS_and_related_classes___________________________________ = 0


def test_class_ap():
    im, hdr = get_test_image()

    # Test constructor, both masks None (default, rare):
    c = image.Ap(im, xy_center=(1476.3, 1243.7), cutout_radius=25.2,
                     foreground_mask=None, background_mask=None)
    assert c.is_valid == True
    assert c.is_pristine == True
    assert np.array_equal(im, c.parent)
    assert c.xy_center == (1476.3, 1243.7)
    assert c.cutout_radius == 26
    assert c.input_foreground_mask is None
    assert c.input_background_mask is None
    assert c.messages == []
    assert c.is_all_within_parent == True
    assert c.x_center == 1476
    assert c.y_center == 1244
    assert c.x_raw_low == c.x_offset == 1450
    assert c.x_raw_high == 1502
    assert c.y_raw_low == c.y_offset == 1218
    assert c.y_raw_high == 1270
    assert c.shape == (53, 53)
    assert np.max(c.data) == 1065
    assert np.min(c.data) == 208
    assert np.sum(c.data) == 754230
    assert c.foreground_mask.shape == c.shape
    assert c.background_mask.shape == c.shape
    assert c.pixel_count == 53 * 53
    assert c.foreground_pixel_count == c.pixel_count
    assert c.background_pixel_count == 0
    assert c.mask_overlap_pixel_count == 0
    assert np.array_equal(c.foreground_mask, np.full_like(c.data, False, dtype=np.bool))
    assert np.array_equal(c.background_mask, np.full_like(c.data, True, dtype=np.bool))
    del c

    # Test constructor, with only foreground mask given (background mask is derived):
    x_offset = 1476 - 26
    y_offset = 1243 - 26
    fg_mask = image.make_circular_mask(53, (1470 - x_offset, 1240 - y_offset), radius=10)
    c = image.Ap(im, xy_center=(1476.3, 1243.7), cutout_radius=25.2,
                     foreground_mask=fg_mask, background_mask=None)
    assert c.is_valid == True
    assert c.is_pristine == True
    assert c.is_all_within_parent == True
    assert np.array_equal(c.foreground_mask, fg_mask)
    assert np.array_equal(c.background_mask, np.logical_not(c.foreground_mask))
    assert c.foreground_pixel_count == 317
    assert c.background_pixel_count == 53 * 53 - c.foreground_pixel_count
    assert c.mask_overlap_pixel_count == 0
    del c

    # Test constructor, with both masks given (masks overlap):
    x_offset = 1476 - 26
    y_offset = 1243 - 26
    fg_mask = image.make_circular_mask(53, (1470 - x_offset, 1240 - y_offset), radius=10)
    bg_mask = np.logical_not(image.make_circular_mask(53, (1480 - x_offset, 1242 - y_offset), radius=12))
    c = image.Ap(im, xy_center=(1476.3, 1243.7), cutout_radius=25.2,
                     foreground_mask=fg_mask, background_mask=bg_mask)
    assert c.is_valid == True
    assert c.is_pristine == True
    assert c.is_all_within_parent == True
    assert np.array_equal(c.foreground_mask, fg_mask)
    assert np.array_equal(c.background_mask, bg_mask)
    assert c.foreground_mask.shape == c.shape
    assert c.background_mask.shape == c.shape
    assert c.foreground_pixel_count == 317
    assert c.background_pixel_count == 2368
    assert c.mask_overlap_pixel_count == 156


def test_class_ap_net_flux():
    """ Test Ap.net_flux(). """
    # TODO: adjust this for new .calc_background_value():
    ap = make_standard_ap_object()
    background_adjusted_flux, flux_stddev, background_level, background_stddev = ap.net_flux(gain=1.57)
    assert background_adjusted_flux == pytest.approx(31481, abs=2)
    assert flux_stddev == pytest.approx(268.4, abs=0.1)
    assert background_level == pytest.approx(257, abs=1)
    assert background_stddev == pytest.approx(14.24, abs=0.1)


def test_class_ap_centroid():
    """ Test Ap class centroid facility. """
    # TODO: adjust this for new .calc_background_value():
    ap = make_offcenter_ap_object()
    x_centroid, y_centroid = ap.xy_centroid
    assert x_centroid == pytest.approx(1476.022, abs=0.005)
    assert y_centroid == pytest.approx(1243.616, abs=0.005)

    # Now, test with cutout centered nearer centroid (but masks unchanged from above code):
    xy_center = 1476, 1244
    ap2 = make_test_ap_object(xy_center)
    x_centroid, y_centroid = ap2.xy_centroid
    assert x_centroid == pytest.approx(1476.249, abs=0.005)
    assert y_centroid == pytest.approx(1243.337, abs=0.005)

    # And now, test with cutout centered even nearer centroid (but masks unchanged from above code):
    xy_center = 1476.25, 1243.34
    ap3 = make_test_ap_object(xy_center)
    x_centroid, y_centroid = ap3.xy_centroid
    assert x_centroid == pytest.approx(1476.278, abs=0.005)
    assert y_centroid == pytest.approx(1243.254, abs=0.005)


def test_class_ap_make_new_object():
    # Test .make_new_object():
    ap = make_standard_ap_object()
    x_centroid, y_centroid = ap.xy_centroid
    assert x_centroid == pytest.approx(1476.278, abs=0.01)
    assert y_centroid == pytest.approx(1243.254, abs=0.01)

    # Make new Ap object from old one, masks retained.
    new_xy_center = 1473, 1247
    new_ap = ap.make_new_object(new_xy_center)
    assert new_ap.is_valid
    assert new_ap.is_pristine
    assert np.array_equal(new_ap.foreground_mask, ap.foreground_mask)
    assert np.array_equal(new_ap.foreground_mask, ap.foreground_mask)
    assert x_centroid == pytest.approx(1476.278, abs=0.005)
    assert y_centroid == pytest.approx(1243.254, abs=0.005)


def test_class_ap_recenter():
    # Test .recenter() (to test algorithm only; .recenter() is not very practical for fixed masks):
    # TODO: write these tests. *****************
    pass


def test_class_pointsourceap():
    im, hdr = get_test_image()

    # Test constructor with all parms specified:
    # TODO: adjust this for new .calc_background_value():
    c = image.PointSourceAp(im, xy_center=(1476, 1243), foreground_radius=10, gap=6, background_width=5)
    assert c.foreground_pixel_count == 317
    assert c.background_pixel_count == 576
    x_centroid, y_centroid = c.xy_centroid
    assert x_centroid == pytest.approx(1476.254, abs=0.005)
    assert y_centroid == pytest.approx(1243.278, abs=0.005)

    # Test .recenter():
    # TODO: write these tests. *****************


_____HELPER_functions_________________________________________________ = 0


def get_test_image():
    """ Return image array for use in above tests. """
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
    hdu0 = apyfits.open(fullpath)[0]
    return hdu0.data, hdu0.header


def make_test_ap_object(xy_center):
    """ So that we don't have to keep making this over and over; so that we have a standard object
        against which to test.
    :param xy_center: target (x,y) center of new ap object, in parent pixels. [2-tuple of floats]
    :return: standard test object with concentric circular masks. [Old_Ap class object]
    """
    im, hdr = get_test_image()
    cutout_radius = 22
    foreground_radius = 10
    gap = 6
    background_width = 5
    cutout_size = 2 * cutout_radius + 1
    xy_cutout_center = cutout_radius, cutout_radius
    radius_inside = foreground_radius + gap
    radius_outside = radius_inside + background_width
    fg_mask = image.make_circular_mask(cutout_size, xy_cutout_center, radius=foreground_radius)
    inner_bg_mask = np.logical_not(image.make_circular_mask(cutout_size, xy_cutout_center, radius_inside))
    outer_bg_mask = image.make_circular_mask(cutout_size, xy_cutout_center,
                                             radius=radius_outside)
    bg_mask = np.logical_or(inner_bg_mask, outer_bg_mask)
    ap_object = image.Ap(im, xy_center=xy_center, cutout_radius=cutout_radius,
                             foreground_mask=fg_mask, background_mask=bg_mask)
    return ap_object


def make_standard_ap_object():
    """ So that we don't have to keep making this over and over; so that we have a standard object
        against which to test.
    :return: standard test object with concentric circular masks. [Ap class object]
    """
    xy_center = 1476, 1243  # near flux centroid.
    return make_test_ap_object(xy_center)


def make_offcenter_ap_object():
    """ So that we don't have to keep making this over and over; so that we have a standard object
        against which to test.
    :return: standard test object with concentric circular masks. [Ap class object]
    """
    xy_center = 1473, 1247  # a few pixels from flux centroid.
    return make_test_ap_object(xy_center)


def test_make_gaussian_ap_object():
    """ So that we don't have to keep making this over and over; so that we have a standard object
        against which to test.
    :return: standard 2-D gaussian test object with concentric circular masks. [Ap class object]
    """
    from astropy.table import Table
    table = Table()
    table['amplitude'] = [100]
    table['x_mean'] = [25]
    table['y_mean'] = [25]
    table['x_stddev'] = [2]
    table['y_stddev'] = [2]
    table['theta'] = np.radians(np.array([0.]))
    from photutils.datasets import make_gaussian_sources_image
    shape = (100, 100)
    parent = make_gaussian_sources_image(shape, table)
    ap = image.PointSourceAp(parent, (25, 25), 10, 6, 5)
    from photutils.morphology import data_properties
    dp = data_properties(ap.data, ap.foreground_mask)
    return ap
