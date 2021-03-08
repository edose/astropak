__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# Extenal packages:
import pandas as pd
import pytest

# Author's packages:
import astropak.legacy
from astropak import image

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")


def test_classes_image_and_aperture():
    test_rel_directory = '$data_for_test'

    # Open FITS file with known extension:
    given_filename = 'CE Aur-0001-V.fts'
    fits_obj = image.FITS(TEST_TOP_DIRECTORY, rel_directory=test_rel_directory, filename=given_filename)
    im = astropak.legacy.Image(fits_obj)
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
    im = astropak.legacy.Image(fits_obj, aperture_radii_pixels=ap_radii)
    assert im.fits.object == 'CE Aur'
    assert im.aperture_radii_pixels == ap_radii
    im.add_aperture('dummy_1', 1523, 1011)  # star near image center, no punches.
    results = im.results_from_aperture('dummy_1')
    assert (results['r_disc'], results['r_inner'], results['r_outer']) == ap_radii
    assert results['n_disc_pixels'] == pytest.approx(316, abs=2)