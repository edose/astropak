__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"


# Python core packages:
import os
import datetime

# External packages:
import pytest
import pandas as pd

# TARGET TEST MODULE:
import astropak.web as web


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")

DSW_SITE_DICT = {'name': 'DSW',
                 'mpc code': 'V28',
                 'longitude': 254.34647,
                 'latitude': 35.3311,
                 'elevation': 2210.0,
                 'coldest date': '01-25',
                 'Extinctions:': {'Clear': (0.16, 0.14),
                                  'I': (0.11, 0.08)}}


def test_make_df_mpes_one_page():
    # Normal case, with file of representative MPES html:
    file_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'MPES_page_example.txt')
    df = web.make_df_mpes_one_page(file_fullpath=file_fullpath)

    # Normal case, web page of MPES html:
    # mp_list = [234, 345, 10111] + [i + 501 for i in range(50)]
    # site_dict = DSW_SITE_DICT
    # utc_start = datetime.datetime(2020, 11, 3, 0, 0, 0, tzinfo=datetime.timezone.utc)
    # df = web.make_df_mpes_one_page(mp_list, site_dict, utc_start)

    iiii = 4


def test_make_df_mpes():
    mp_list = [i + 1 for i in range(142)]
    site_dict = DSW_SITE_DICT
    utc_start = datetime.datetime(2020, 11, 3, 0, 0, 0, tzinfo=datetime.timezone.utc)
    df_mpes = web.make_df_mpes(mp_list, site_dict, utc_start)
    iiii = 4


def test_try_lcdb():
    mp_number = 4954  # has color info.
    site_dict = DSW_SITE_DICT
    utc_start = '2020-11-02'
    ci_table = web.try_lcdb(mp_number, site_dict, utc_start)
    return ci_table


def test_class_lcdb_colors():
    fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'LC_COLORINDEX_PUB.TXT')
    c = web.LCDB_Colors(fullpath)
    assert c.has_id(5) == True
    assert c.has_id(18) == False
    assert c.color_count(5) == 2
    assert c.color_count(18) == 0
    assert c.color_count(21) == 5
    assert c.color_count('2009 BO64') == 4
    assert sum([c.color_count(i) for i in [174567, '1996 TK66']]) == 10 + 7
    assert c.sloan_count(21) == 0
    assert c.sloan_count('2006 QJ181') == 2
    assert c.jc_count('NOT AN ID') == 0
    assert c.jc_count(40035) == 4
