__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pandas as pd
import pytest

# Other astropak modules:

# TARGET TEST MODULE:
from astropak import catalogs

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")


def test_class_Refcat2():
    """ Regression testing only; tested originally and unchanged since. """
    test_rel_directory = '$data_for_test'

    # Test normal case, very small sky rectangle:
    cat = catalogs.Refcat2(ra_deg_range=(240.2, 240.3), dec_deg_range=(13.1, 13.2))
    assert cat.n_stars() == len(cat.df_raw) == 8
    assert cat.df_raw.loc[149, 'g'] == pytest.approx(14.30500, abs=0.00001)
    assert len(cat.df_selected) == len(cat.df_raw)
    assert cat.df_selected.loc[149, 'g'] == cat.df_raw.loc[149, 'g']
    cat.select_max_r_mag(15.5)
    cat.select_min_r_mag(13)
    assert len(cat.df_raw) == 8
    assert len(cat.df_selected) == 3
    assert cat.epoch == catalogs.ATLAS_REFCAT2_EPOCH_UTC

    cat = catalogs.Refcat2.from_fits_file(os.path.join(TEST_TOP_DIRECTORY, test_rel_directory),
                                          'CE Aur-0001-V.fts')
    assert cat.n_stars() == len(cat.df_raw) == 411
    cat.select_max_r_mag(15.5)
    cat.select_min_r_mag(13)
    assert len(cat.df_raw) == 411
    assert cat.n_stars() == len(cat.df_selected) == 223



