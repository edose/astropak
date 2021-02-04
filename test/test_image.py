__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytest
import astropy.io.fits as apyfits

# Other astropak modules:
from astropak.util import RaDec, dec_as_degrees, ra_as_degrees

# TARGET TEST MODULE:
from astropak import image

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")


def test_class_fits():
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


def test_classes_image_and_aperture():
    test_rel_directory = '$data_for_test'

    # Open FITS file with known extension:
    given_filename = 'CE Aur-0001-V.fts'
    fits_obj = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename=given_filename)
    im = image.Image(fits_obj)
    assert im.fits.object == 'CE Aur'
    assert im.top_directory == TEST_TOP_DIRECTORY
    assert im.rel_directory == test_rel_directory
    assert im.xsize, im.ysize == fits_obj.image_xy.shape  # .shape is in nrows, ncols
    # Image dimensions are x,y == *image* cols,rows, the reverse of numpy storage.
    # Images are zero based; [0,0] -> upper-left, [n, 0] is on top edge of *image* (not of storage).

    assert im.image.shape == (3072, 2047)
    assert im.image[0, 0] == 275        # upper-left corner
    assert im.image[0, 2046] == 180     # lower-left corner
    assert im.image[3071, 2046] == 265  # lower-right corner
    assert im.image[3071, 0] == 285     # upper-right corner

    # Aperture very simple case: near image center, no punches or interfering signals:
    assert im.aperture_radii_pixels == (9, 15, 20)  # Verify radii, first--tests below depend on them.
    im.add_aperture('dummy_1', 1523, 1011)  # star near image center, no punches.
    assert len(im.apertures) == 1
    this_ap = im.apertures['dummy_1']
    assert this_ap.x_centroid == pytest.approx(1524.784, abs=0.005)
    results = im.results_from_aperture('dummy_1')
    assert results['x_centroid'] == this_ap.x_centroid
    assert results['fwhm'] == pytest.approx(6.22, abs=0.02)
    assert set(results.index) == set(['r_disc', 'r_inner', 'r_outer', 'n_disc_pixels',
                                      'n_annulus_pixels', 'net_flux', 'net_flux_sigma',
                                      'annulus_flux', 'annulus_flux_sigma',
                                      'x_centroid', 'y_centroid', 'fwhm',
                                      'x1024', 'y1024', 'vignette', 'sky_bias', 'max_adu'])

    # Aperture case: near image center, two punches:
    im.add_aperture('dummy_2', 1535, 979)
    df_punches = pd.DataFrame({'StarID': 'dummy_2',
                               'dNorth': [-11.1, +9.6],
                               'dEast': [0.0, +3.4]})
    im.add_punches(df_punches)
    assert len(im.apertures) == 2
    this_ap = im.apertures['dummy_2']
    assert [this_ap.x_centroid, this_ap.y_centroid] == pytest.approx([1534.456, 978.697], abs=0.2)
    results = im.results_from_aperture('dummy_2')
    assert results['x_centroid'] == this_ap.x_centroid
    assert results['fwhm'] == pytest.approx(6.15, abs=0.2)

    # Aperture case: far from image center, one punch:
    im.add_aperture('dummy_3', 510, 483)
    df_punches = pd.DataFrame({'StarID': ['dummy_2', 'trash', 'dummy_3'],
                               'dNorth': [-11.1, -99.9, +8.9],
                               'dEast': [0.0, +99.9, 0.0]})  # verify safety of non-relevant rows.
    im.add_punches(df_punches)
    assert len(im.apertures) == 3
    this_ap = im.apertures['dummy_3']
    assert [this_ap.x_centroid, this_ap.y_centroid] == pytest.approx([505.53, 481.35], abs=0.2)
    results = im.results_from_aperture('dummy_3')
    assert results['annulus_flux'] == pytest.approx(252.7, abs=1)
    assert results['annulus_flux_sigma'] == pytest.approx(15.8, abs=0.5)
    assert results['fwhm'] == pytest.approx(6.6, abs=0.1)
    assert results['max_adu'] == 441
    assert results['n_annulus_pixels'] == pytest.approx(459, abs=1)
    assert results['n_disc_pixels'] == pytest.approx(255, abs=1)
    assert results['net_flux'] == pytest.approx(6931, abs=1)
    assert results['net_flux_sigma'] == pytest.approx(320, abs=1)
    assert results['r_disc'] == 9
    assert results['r_inner'] == 15
    assert results['r_outer'] == 20
    assert results['sky_bias'] == pytest.approx(0.62, abs=0.1)
    assert results['vignette'] == pytest.approx(results['x1024']**2 + results['y1024']**2, abs=0.01)
    assert results['x1024'] == pytest.approx(-1.006, abs=0.01)
    assert results['x_centroid'] == pytest.approx(505.6, abs=0.1)
    assert results['y1024'] == pytest.approx(-0.529, abs=0.01)
    assert results['y_centroid'] == pytest.approx(481.3, abs=0.1)
    assert results['y_centroid'] == this_ap.y_centroid  # verify equivalence

    # Test effect of aperture_radii_pixels:
    given_filename = 'CE Aur-0001-V.fts'
    ap_radii = (10, 12, 14)
    fits_obj = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory,
                          filename=given_filename)
    im = image.Image(fits_obj, aperture_radii_pixels=ap_radii)
    assert im.fits.object == 'CE Aur'
    assert im.aperture_radii_pixels == ap_radii
    im.add_aperture('dummy_1', 1523, 1011)  # star near image center, no punches.
    results = im.results_from_aperture('dummy_1')
    assert (results['r_disc'], results['r_inner'], results['r_outer']) == ap_radii
    assert results['n_disc_pixels'] == pytest.approx(316, abs=2)


def test_fits__xy_from_radec():
    from astropak.util import RaDec
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


_____IMAGE_and_GEOMETRY_SUPPORT____________________________________ = 0


def test_distance_to_line():  # OK 2021-01-28.
    assert image.distance_to_line((3, 4), (-1, 7), (-1, -33)) == 4.0  # horizontal line.
    assert image.distance_to_line((35, 4), (-100, 7), (100, 7)) == 3.0  # verical line.
    assert image.distance_to_line((15, 12), (-12, 43), (13, -17)) == 13.0  # normal within line segment.
    assert image.distance_to_line((15, 12), (-12, 43), (23, -41)) == 13.0  # normal outside line segment.
    assert image.distance_to_line((3, 4), (-1, 7), (-1, -33), dist_ab=40.0) == 4.0


def test_make_circular_mask():  # OK 2021-02-03.
    cm = image.make_circular_mask(25, (12.2, 15.5), 8)
    assert isinstance(cm, np.ndarray)
    assert cm.dtype == np.bool
    assert cm.shape == (25, 25)
    assert np.sum(cm == False) == 200  # number of valid pixels.
    assert not np.any(cm[12:18, 8:14])


def test_make_pill_mask():  # OK 2021-01-28.
    pm = image.make_pill_mask(41, (15, 20), (10, 18), 4)
    assert isinstance(pm, np.ndarray)
    assert pm.shape == (41, 41)
    assert np.sum(pm == False) == 92  # number of valid pixels.

    pm = image.make_pill_mask(43, (10, 20), (10, 18), 7)  # vertical motion.
    assert pm.shape == (43, 43)
    assert np.sum(pm == False) == 179  # number of valid pixels.

    pm = image.make_pill_mask(60, (13, 20), (28, 20), 9)  # horizontal motion, increasing x.
    assert pm.shape == (60, 60)
    assert np.sum(pm == False) == 538  # number of valid pixels.

    pm_reverse = image.make_pill_mask(60, (28, 20), (13, 20), 9)  # horizontal motion, decreasing x.
    assert np.array_equal(pm_reverse, pm)  # doesn't matter which point is the start, which is end.


def test_calc_background_value():
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
    hdu0 = apyfits.open(fullpath)[0]
    im = hdu0.data

    data = im.image[1218:1268, 1451:1501]  # nb: (y,x) per numpy.
    # TODO: Add tests here.
    #  mask = None


def test_class_cutout_constructors():
    # Rewritten 2021-01-30 for new two-mask version.
    # Setup:
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
    hdu0 = apyfits.open(fullpath)[0]
    im = hdu0.data

    # Test main constructor without masks:
    c = image.Cutout(im, (1476.3, 1243.7), 25.2)
    assert c.is_valid
    assert c.xy_center == (1476.3, 1243.7)
    assert c.input_radius == 25.2
    assert c.input_foreground_mask is None
    assert c.input_background_mask is None
    assert c.messages == []
    assert c.is_pristine == True
    assert c.is_valid == True
    assert c.is_all_within_parent == True
    assert c.x_cutout_center == 1476
    assert c.y_cutout_center == 1244
    assert c.radius == 26
    assert c.parent.shape == im.shape
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

    # Test constructor with: explicit cutout-shaped foreground mask, no background mask:
    c = image.Cutout(im, (1476.3, 1243.7), 25.2)  # no masks given.
    fg_mask = np.full_like(c.data, True, dtype=np.bool)
    fg_mask[2, 3] = False
    cm = image.Cutout(im, (1476.3, 1243.7), 25.2, foreground_mask=fg_mask)
    assert cm.is_pristine == True
    assert cm.is_valid == True
    assert cm.foreground_mask.shape == cm.background_mask.shape == cm.shape
    assert np.array_equal(cm.foreground_mask, fg_mask)
    expected_bg_mask = np.logical_not(cm.foreground_mask)
    assert np.array_equal(cm.background_mask, expected_bg_mask)
    del c, cm

    # Test constructor with: explicit cutout-shaped foreground mask and background mask:
    c = image.Cutout(im, (1476.3, 1243.7), 25.2)  # no masks given.
    fg_mask = np.full_like(c.data, True, dtype=np.bool)
    fg_mask[2, 3] = False
    bg_mask = np.full_like(c.data, True, dtype=np.bool)
    bg_mask[5, 4] = False
    cmm = image.Cutout(im, (1476.3, 1243.7), 25.2, foreground_mask=fg_mask, background_mask=bg_mask)
    assert cmm.is_pristine == True
    assert cmm.is_valid == True
    assert cmm.foreground_mask.shape == cmm.background_mask.shape == cmm.shape
    assert np.array_equal(cmm.foreground_mask, fg_mask)
    assert np.array_equal(cmm.background_mask, bg_mask)
    del c, cmm

    # Test constructor with: explicit parent(image)-shaped foreground mask, no background mask:
    c = image.Cutout(im, (1476.3, 1243.7), 25.2)  # no masks given.
    fg_mask = np.full_like(im, True, dtype=np.bool)
    fg_mask[1240, 1479] = False  # [y, x] per numpy
    cp = image.Cutout(im, (1476.3, 1243.7), 25.2, foreground_mask=fg_mask)
    assert cp.is_pristine == True
    assert cp.is_valid == True
    assert cp.foreground_mask.shape == cp.background_mask.shape == cp.shape
    assert np.sum(cp.foreground_mask) == cp.background_mask.size - 1
    assert cp.foreground_mask[22, 29] == False  # [y, x] per numpy
    expected_bg_mask = np.logical_not(cp.foreground_mask)
    assert np.array_equal(cp.background_mask, expected_bg_mask)
    del c, cp

    # Test constructor with: explicit parent(image)-shaped foreground mask and background mask:
    c = image.Cutout(im, (1476.3, 1243.7), 25.2)  # no masks given.
    fg_mask = np.full_like(im, True, dtype=np.bool)
    fg_mask[1240, 1479] = False  # [y, x] per numpy
    bg_mask = np.full_like(im, True, dtype=np.bool)
    bg_mask[1244, 1474] = False
    cpp = image.Cutout(im, (1476.3, 1243.7), 25.2, foreground_mask=fg_mask, background_mask=bg_mask)
    assert cpp.is_pristine == True
    assert cpp.is_valid == True
    assert cpp.foreground_mask.shape == cpp.background_mask.shape == cpp.shape
    assert np.sum(cpp.foreground_mask) == cpp.foreground_mask.size - 1
    assert np.sum(cpp.background_mask) == cpp.background_mask.size - 1
    assert cpp.foreground_mask[22, 29] == False  # [y, x] per numpy
    assert cpp.background_mask[26, 24] == False  # [y, x] per numpy
    del c, cpp

    # Test constructor with: circular mask and mask radius:
    ccr = image.Cutout.circular_mask_and_radius(im, (1476.3, 1243.3), cutout_radius=25.2,
                                                foreground_mask_radius=10.7, gap=2)
    assert ccr.is_valid
    assert ccr.is_pristine
    assert ccr.xy_center == (1476.3, 1243.3)
    assert ccr.radius == 26
    assert ccr.x_raw_low == ccr.x_offset == 1450
    assert ccr.x_raw_high == 1502
    assert ccr.y_raw_low == ccr.y_offset == 1217
    assert ccr.y_raw_high == 1269
    assert ccr.shape == (53, 53)
    assert ccr.x_cutout_center == 1476
    assert ccr.y_cutout_center == 1243
    assert isinstance(ccr.foreground_mask, np.ndarray)
    assert isinstance(ccr.background_mask, np.ndarray)
    assert np.sum(ccr.foreground_mask) == 2452
    assert np.sum(ccr.background_mask) == 504
    del ccr

    # Test constructor with: circular mask and margin:
    ccm = image.Cutout.circular_mask_and_margin(im, (1476.3, 1243.3), margin=9,
                                                foreground_mask_radius=10.7, gap=2)
    assert ccm.is_valid
    assert ccm.data.shape == (45, 45)
    assert ccm.xy_center == (1476.3, 1243.3)
    assert ccm.x_raw_low == ccm.x_offset == 1454
    assert ccm.x_raw_high == 1498
    assert ccm.y_raw_low == ccm.y_offset == 1221
    assert ccm.y_raw_high == 1265
    assert isinstance(ccm.foreground_mask, np.ndarray)
    assert isinstance(ccm.background_mask, np.ndarray)
    assert np.sum(ccm.foreground_mask) == 1668
    assert np.sum(ccm.background_mask) == 504
    del ccm

    # Test constructor with: pill mask and radius, no gap:
    cpr = image.Cutout.pill_mask_and_radius(im, xy_start=(1474.3, 1241.3), xy_end=(1478.3, 1245.3),
                                            cutout_radius=25.2, foreground_mask_radius=10.7)  # so, gap=0
    assert cpr.is_valid
    assert cpr.is_pristine
    assert cpr.data.shape == (53, 53)
    assert cpr.xy_center == (1476.3, 1243.3)
    assert cpr.x_raw_low == cpr.x_offset == 1450
    assert cpr.x_raw_high == 1502
    assert cpr.y_raw_low == cpr.y_offset == 1217
    assert cpr.y_raw_high == 1269
    assert isinstance(cpr.foreground_mask, np.ndarray)
    assert isinstance(cpr.background_mask, np.ndarray)
    assert np.sum(cpr.foreground_pixel_count) == cpr.data.shape[0]**2 - np.sum(cpr.foreground_mask) == 481
    assert np.sum(cpr.background_pixel_count) == cpr.data.shape[0]**2 - np.sum(cpr.background_mask) == 2328
    assert np.array_equal(cpr.background_mask, np.logical_not(cpr.foreground_mask))
    del cpr

    # Test constructor with: pill mask and radius, user-supplied gap:
    cpr2 = image.Cutout.pill_mask_and_radius(im, xy_start=(1474.3, 1241.3), xy_end=(1478.3, 1245.3),
                                             cutout_radius=25.2, foreground_mask_radius=10.7, gap=2)
    assert cpr2.is_valid
    assert cpr2.is_pristine
    assert cpr2.data.shape == (53, 53)
    assert cpr2.x_raw_low == cpr2.x_offset == 1450
    assert cpr2.x_raw_high == 1502
    assert cpr2.y_raw_low == cpr2.y_offset == 1217
    assert cpr2.y_raw_high == 1269
    assert isinstance(cpr2.foreground_mask, np.ndarray)
    assert isinstance(cpr2.background_mask, np.ndarray)
    assert cpr2.foreground_pixel_count == cpr2.data.shape[0]**2 - np.sum(cpr2.foreground_mask) == 481
    assert cpr2.background_pixel_count == cpr2.data.shape[0]**2 - np.sum(cpr2.background_mask) == 2165
    assert not np.array_equal(cpr2.background_mask, np.logical_not(cpr2.foreground_mask))
    del cpr2

    # Test constructor with: pill mask and margin:
    cpm = image.Cutout.pill_mask_and_margin(im, (1474.3, 1241.3), (1478.3, 1245.3), margin=9.1,
                                            foreground_mask_radius=10.7, gap=2)
    assert cpm.is_valid
    assert cpm.is_pristine
    assert cpm.data.shape == (49, 49)
    assert cpm.x_raw_low == cpm.x_offset == 1452
    assert cpm.x_raw_high == 1500
    assert cpm.y_raw_low == cpm.y_offset == 1219
    assert cpm.y_raw_high == 1267
    assert isinstance(cpm.foreground_mask, np.ndarray)
    assert isinstance(cpm.background_mask, np.ndarray)
    assert cpm.foreground_pixel_count == cpm.data.shape[0]**2 - np.sum(cpm.foreground_mask) == 481
    assert cpm.background_pixel_count == cpm.data.shape[0]**2 - np.sum(cpm.background_mask) == 1757
    assert not np.array_equal(cpm.background_mask, np.logical_not(cpm.foreground_mask))
    del cpm


def test_class_cutout_flux():
    # Setup:
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
    hdu0 = apyfits.open(fullpath)[0]
    im = hdu0.data

    c = image.Cutout.circular_mask_and_radius(im, (1476, 1243), cutout_radius=25,
                                              foreground_mask_radius=10, gap=6)
    flux, flux_stddev, sky, sky_stddev = c.flux(use_foreground_mask=True, use_background_mask=True,
                                                gain=1.57)  # NORMAL aperture photometry case.
    assert flux == pytest.approx(31481, abs=10)
    assert flux_stddev == pytest.approx(268.3, abs=1)
    assert sky == pytest.approx(257, abs=1)
    assert sky_stddev == pytest.approx(14.02, 0.05)

    flux, flux_stddev, sky, sky_stddev = c.flux(use_foreground_mask=True, use_background_mask=False,
                                                gain=1.57)  # RARE ap. photometry w/o backgd subtraction.
    assert flux == pytest.approx(112950, abs=10)
    assert flux_stddev == pytest.approx(268.2, abs=1)
    assert sky == 0.0
    assert sky_stddev == 0.0

    flux, flux_stddev, sky, sky_stddev = c.flux(use_foreground_mask=False, use_background_mask=True,
                                                gain=1.57)  # ABNORMAL case.
    assert flux == pytest.approx(619349, abs=10)
    assert flux_stddev == pytest.approx(668.1, abs=1)
    assert sky == pytest.approx(257, abs=1)
    assert sky_stddev == pytest.approx(14.02, 0.05)

    flux, flux_stddev, sky, sky_stddev = c.flux(use_foreground_mask=False, use_background_mask=False,
                                                gain=1.57)  # simple ADU sum over entire Cutout.
    assert flux == pytest.approx(700818, abs=10)
    assert flux_stddev == pytest.approx(668.1, abs=1)
    assert sky == 0.0
    assert sky_stddev == 0.0


def test_class_cutout_calc_centroid():
    # Setup:
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'CE Aur-0001-V.fts')
    hdu0 = apyfits.open(fullpath)[0]
    im = hdu0.data

    # ========== Test .calc_centroid():
    cc = image.Cutout.circular_mask_and_radius(im, (1476, 1243), cutout_radius=25,
                                               foreground_mask_radius=10, gap=6)
    assert cc.is_valid
    assert cc.xy_center == (1476, 1243)
    x_centroid, y_centroid = cc.xy_centroid
    assert x_centroid == pytest.approx(1476.254, abs=0.005)
    assert y_centroid == pytest.approx(1243.278, abs=0.005)

    cc2 = image.Cutout.circular_mask_and_radius(im, (1480, 1247), cutout_radius=20,
                                                foreground_mask_radius=10, gap=6)
    assert cc2.is_valid
    assert cc2.xy_center == (1480, 1247)
    x_centroid, y_centroid = cc2.xy_centroid
    assert x_centroid == pytest.approx(1476.58, abs=0.1)
    assert y_centroid == pytest.approx(1243.54, abs=0.1)

    iiii = 4
