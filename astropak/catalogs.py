__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from datetime import datetime, timezone
from math import floor, cos, pi

# External packages:
import pandas as pd

# EVD packages:
from .image import FITS
from .legacy import Image

# ATLAS refcat2 constants:
ATLAS_REFCAT2_DIRECTORY = 'D:/Astro/Catalogs/ATLAS-refcat2/mag-0-16/'
RP1_LIMIT = 9  # arcseconds; closeness limit for flux = 0.1 * star flux
R1_LIMIT = 14   # arcseconds; closeness limit for flux = 1 * star flux
R10_LIMIT = 20  # arcseconds; closeness limit for flux = 10 * star flux
ATLAS_REFCAT2_EPOCH_UTC = (datetime(2015, 1, 1) +
                           (datetime(2016, 1, 1) - datetime(2015, 1, 1)) / 2.0)\
    .replace(tzinfo=timezone.utc)  # refcat2 proper-motion epoch is 2015.5.

# TODO: Move these to new ref.py.
DAYS_PER_YEAR_NOMINAL = 365.25
DEGREES_PER_RADIAN = 180.0 / pi


MATCH_TOLERANCE_ARCSEC = 4  # default tolerance to match stars across catalogs.


ATLAS_REFCAT2_____________________________________________________ = 0


class Refcat2:
    """ Container for ATLAS refcat2 star data within a RA,Dec rectangle."""
    def __init__(self, ra_deg_range=(None, None), dec_deg_range=(None, None),
                 sort_ra=True, directory=ATLAS_REFCAT2_DIRECTORY):
        """ Basic constructor. Other constructors call this one.
        :param ra_deg_range: (min RA, max RA) tuple. Handles zero-crossing gracefully [2-tuple of floats].
        :param dec_deg_range: (min Dec, max Dec) tuple [2-tuple of floats].
        :param sort_ra: True iff returned star data are to be sorted in RA order [boolean].
        :param directory: where ATLAS refcat2 is found [string].
        Columns in core dataframe: RA_deg, Dec_deg, PM_ra, dPM_ra, PM_dec, dPM_dec,
               G_gaia, dG_gaia, BP_gaia, dBP_gaia, RP_gaia, dRP_gaia,
               T_eff, dupvar, RP1, R1, R10, g, dg, r, dr, i, di; index=unique integers.
        """
        ra_spec_first = int(floor(ra_deg_range[0])) % 360  # will always be in [0, 360).
        ra_spec_last = int(floor(ra_deg_range[1])) % 360   # "
        if ra_spec_last < ra_spec_first:
            ra_spec_last += 360
        dec_spec_first = int(floor(dec_deg_range[0]))
        dec_spec_first = max(dec_spec_first, -90)
        dec_spec_last = int(floor(dec_deg_range[1]))
        dec_spec_last = min(dec_spec_last, 89)
        # print('RA: ', str(ra_spec_first), str((ra_spec_last % 360) + 1))
        # print('Dec:', str(dec_spec_first), str(dec_spec_last))
        df_list = []
        for ra_spec in range(ra_spec_first, ra_spec_last + 1):
            for dec_spec in range(dec_spec_first, dec_spec_last + 1):
                df_degsq = read_one_refcat2_sqdeg(directory, ra_spec % 360, dec_spec)
                # print('From:', str(ra_spec % 360), str(dec_spec), ' -> ', str(len(df_degsq)), 'rows.')
                df_list.append(df_degsq)
        df = pd.DataFrame(pd.concat(df_list, ignore_index=True))  # new index of unique integers
        print('\nRefcat2: begin with', str(len(df)), 'stars.')

        # Trim dataframe based on user's actual limits on RA and Dec:
        ra_too_low = (df['RA_deg'] < ra_deg_range[0]) & (df['RA_deg'] >= ra_spec_first)
        ra_too_high = (df['RA_deg'] > ra_deg_range[1]) & (df['RA_deg'] <= (ra_spec_last % 360) + 1)
        dec_too_low = df['Dec_deg'] < dec_deg_range[0]
        dec_too_high = df['Dec_deg'] > dec_deg_range[1]
        # print(str(sum(ra_too_low)), str(sum(ra_too_high)), str(sum(dec_too_low)), str(sum(dec_too_high)))
        radec_outside_requested = ra_too_low | ra_too_high | dec_too_low | dec_too_high
        df = df[~radec_outside_requested]
        print('Refcat2: RADec-trimmed to', str(len(df)), 'stars.')

        # Add columns for synthetic B-V color & synthetic APASS (~Sloan) R magnitude:
        df.loc[:, 'BminusV'] = [0.830 * g - 0.803 * r for (g, r) in zip(df['g'], df['r'])]
        df.loc[:, 'APASS_R'] = [0.950 * r + 0.05 * i for (r, i) in zip(df['r'], df['i'])]

        if sort_ra is True:
            self.df_raw = df.copy().sort_values(by='RA_deg')  # in case all are needed (unlikely)
        self.df_selected = self.df_raw.copy()  # the working copy.
        self.epoch = ATLAS_REFCAT2_EPOCH_UTC

    @classmethod
    def from_fits_object(cls, fits_object):
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = fits_object.bounding_ra_dec()
        return cls((ra_deg_min, ra_deg_max), (dec_deg_min, dec_deg_max), sort_ra=True)

    @classmethod
    def from_fits_file(cls, fits_directory, fits_filename):
        fits_object = FITS(fits_directory, '', fits_filename)
        return cls.from_fits_object(fits_object)

    def n_stars(self):
        return len(self.df_selected)

    def selected_columns(self, column_list):
        return self.df_selected.loc[:, column_list].copy()

    def select_min_r_mag(self, min_r_mag):
        rows_to_keep = [(r >= min_r_mag) for r in self.df_selected['r']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_r_mag(self, max_r_mag):
        rows_to_keep = [(r <= max_r_mag) for r in self.df_selected['r']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_g_uncert(self, max_dg):
        rows_to_keep = [(dg <= max_dg) for dg in self.df_selected['dg']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_r_uncert(self, max_dr):
        rows_to_keep = [(dr <= max_dr) for dr in self.df_selected['dr']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_i_uncert(self, max_di):
        rows_to_keep = [(di <= max_di) for di in self.df_selected['di']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_bv_color(self, min_bv=None, max_bv=None):
        above_min = [True if min_bv is None else (bv >= min_bv) for bv in self.df_selected['BminusV']]
        below_max = [True if max_bv is None else (bv <= max_bv) for bv in self.df_selected['BminusV']]
        rows_to_keep = [a and b for (a, b) in zip(above_min, below_max)]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_sloan_ri_color(self, min_sloan_ri=None, max_sloan_ri=None):
        sloan_ri_color = self.df_selected['r'] - self.df_selected['i']
        above_min = [True if min_sloan_ri is None else (ri >= min_sloan_ri) for ri in sloan_ri_color]
        below_max = [True if max_sloan_ri is None else (ri <= max_sloan_ri) for ri in sloan_ri_color]
        rows_to_keep = [a and b for (a, b) in zip(above_min, below_max)]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_dgaia(self):
        rows_to_keep = [(dgaia > 0) for dgaia in self.df_selected['dG_gaia']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def remove_overlapping(self):
        rp1_too_close = pd.Series([False if pd.isnull(rp1) else (rp1 < RP1_LIMIT)
                                   for rp1 in self.df_selected['RP1']])
        r1_too_close = pd.Series([False if pd.isnull(r1) else (r1 < R1_LIMIT)
                                  for r1 in self.df_selected['R1']])
        r10_too_close = pd.Series([False if pd.isnull(r10) else (r10 < R10_LIMIT)
                                   for r10 in self.df_selected['R10']])
        is_overlapping = rp1_too_close | r1_too_close | r10_too_close
        self.df_selected = self.df_selected.loc[list(~is_overlapping), :].copy()

    def update_epoch(self, new_datetime_utc):
        d_years = (new_datetime_utc - ATLAS_REFCAT2_EPOCH_UTC).total_seconds() /\
                  (DAYS_PER_YEAR_NOMINAL * 24 * 3600)
        ra_date = [(ra_epoch + d_years * pm_ra / 3600.0) % 360
                   for (ra_epoch, pm_ra) in zip(self.df_selected['RA_deg'], self.df_selected['PM_ra'])]
        dec_date = [dec_epoch + d_years * pm_dec / 3600.0
                    for (dec_epoch, pm_dec) in zip(self.df_selected['Dec_deg'], self.df_selected['PM_dec'])]
        df_comps_new_date = self.df_selected.copy()
        df_comps_new_date.loc[:, 'RA_deg'] = ra_date
        df_comps_new_date.loc[:, 'Dec_deg'] = dec_date
        self.df_selected = df_comps_new_date
        self.epoch = new_datetime_utc


def read_one_refcat2_sqdeg(directory=ATLAS_REFCAT2_DIRECTORY, ra_deg_min=None, dec_deg_min=None):
    """ From ATLAS refcat2, get all stars within one square degree of sky (within one catalog file).
    :param directory: path to ATLAS refcat2 catalog.
    :param ra_deg_min: lowest RA of square degree needed. [int, or float which will be rounded down]
    :param dec_deg_min:lowest Dec of square degree needed. [int, or float which will be rounded down]
    :return: dataframe of catalog contents, one row per star. [pandas Dataframe]
    """
    ra_deg_int = int(ra_deg_min)    # floor effect (rounds down).
    dec_deg_int = int(dec_deg_min)  # "
    filename = '{:03d}'.format(ra_deg_int) + '{:+03d}'.format(dec_deg_int) + '.rc2'
    fullpath = os.path.join(directory, filename)
    df = pd.read_csv(fullpath, sep=',', engine='python', header=None,
                     skip_blank_lines=True, error_bad_lines=False,
                     usecols=[0, 1, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13,
                              14, 16, 18, 19, 20,
                              21, 22, 25, 26, 29, 30, 33, 34], prefix='col')
    df.columns = ['RA_deg', 'Dec_deg', 'PM_ra', 'dPM_ra', 'PM_dec', 'dPM_dec',
                  'G_gaia', 'dG_gaia', 'BP_gaia', 'dBP_gaia', 'RP_gaia', 'dRP_gaia',
                  'T_eff', 'dupvar', 'RP1', 'R1', 'R10',
                  'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz']
    df['RA_deg'] *= 0.00000001
    df['Dec_deg'] *= 0.00000001
    df['PM_ra'] *= 0.00001    # proper motion in arcsec/year
    df['dPM_ra'] *= 0.00001   # uncert in PM, arcsec/year
    df['PM_dec'] *= 0.00001   # proper motion in arcsec/year
    df['dPM_dec'] *= 0.00001  # uncert in PM, arcsec/year
    df['G_gaia'] *= 0.001  # in magnitudes; dG_gaia remains in millimagnitudes
    df['BP_gaia'] *= 0.001  # in magnitudes; dBP_gaia remains in millimagnitudes
    df['RP_gaia'] *= 0.001  # in magnitudes; dRP_gaia remains in millimagnitudes
    df['RP1'] = [None if rp1 == 999 else rp1 / 10.0 for rp1 in df['RP1']]  # radius in arcseconds
    df['R1'] = [None if r1 == 999 else r1 / 10.0 for r1 in df['R1']]       # "
    df['R10'] = [None if r10 == 999 else r10 / 10.0 for r10 in df['R10']]  # "
    df['g'] *= 0.001  # in magnitudes; dg remains in millimagnitudes
    df['r'] *= 0.001  # in magnitudes; dr remains in millimagnitudes
    df['i'] *= 0.001  # in magnitudes; di remains in millimagnitudes
    df['z'] *= 0.001  # in magnitudes; dz remains in millimagnitudes
    id_prefix = '{:03d}'.format(ra_deg_int) + '{:+03d}'.format(dec_deg_int) + '_'
    id_list = [id_prefix + '{:0>6d}'.format(i + 1) for i in range(len(df))]  # unique in entire catalog.
    df.insert(0, 'CatalogID', id_list)
    print('Refcat2 sqdeg [' + str(ra_deg_int) + ', ' + str(dec_deg_int) + ']: ' + str(len(df)) + ' stars.')
    return df


_____CATALOG_RELATED_FUNCTIONS______________________________ = 0


# TODO: This fn is confused in its signature, usage, and location. E.g., why a dataframe?
def find_matching_comp(df_comps, ra_deg, dec_deg, tolerance_arcseconds=MATCH_TOLERANCE_ARCSEC):
    """ Find ATLAS refcat2 (as df_comps) stars matching input ra_deg, dec_deg; closest if >1 matching.
    :return: Index of matching star in df_comps.
    """
    tol_deg = tolerance_arcseconds / 3600.0
    ra_tol = abs(tol_deg / cos(dec_deg * DEGREES_PER_RADIAN))
    dec_tol = tol_deg
    within_ra = (abs(df_comps['RA_deg'] - ra_deg) < ra_tol) |\
                (abs((df_comps['RA_deg'] + 360.0) - ra_deg) < ra_tol) |\
                (abs(df_comps['RA_deg'] - (ra_deg + 360.0)) < ra_tol)
    within_dec = abs(df_comps['Dec_deg'] - dec_deg) < dec_tol
    within_box = within_ra & within_dec
    if sum(within_box) == 0:
        return None
    elif sum(within_box) == 1:
        return (df_comps.index[within_box])[0]
    else:
        # Here, we choose the *closest* df_comps comp and return its index:
        df_sub = df_comps.loc[list(within_box), ['RA_deg', 'Dec_deg']]
        cos2 = cos(dec_deg) ** 2
        dist2_ra_1 = ((df_sub['RA_deg'] - ra_deg) ** 2) / cos2
        dist2_ra_2 = (((df_sub['RA_deg'] + 360.0) - ra_deg) ** 2) / cos2
        dist2_ra_3 = ((df_sub['RA_deg'] - (ra_deg + 360.0)) ** 2) / cos2
        dist2_ra = [min(d1, d2, d3) for (d1, d2, d3) in zip(dist2_ra_1, dist2_ra_2, dist2_ra_3)]
        dist2_dec = (df_sub['Dec_deg'] - dec_deg) ** 2
        dist2 = [ra2 + dec2 for (ra2, dec2) in zip(dist2_ra, dist2_dec)]
        i = dist2.index(min(dist2))
        return df_sub.index.values[i]

