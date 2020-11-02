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
    mp_list = [i + 1 for i in range(342)]
    site_dict = DSW_SITE_DICT
    utc_start = datetime.datetime(2020, 11, 3, 0, 0, 0, tzinfo=datetime.timezone.utc)
    df_mpes = web.make_df_mpes(mp_list, site_dict, utc_start)
    iiii = 4



