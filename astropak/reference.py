__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

from math import pi, sqrt, log

# NUMERICAL CONSTANTS:
RADIANS_PER_DEGREE = pi / 180.0
DEGREES_PER_RADIAN = 1 / RADIANS_PER_DEGREE        # ca. 57.396
ARCSECONDS_PER_RADIAN = 3600 * DEGREES_PER_RADIAN  # ca. 206265
FWHM_PER_SIGMA = 2.0 * sqrt(2.0 * log(2))          # ca. 2.35482
DAYS_PER_YEAR_NOMINAL = 365.25


# ASTRONOMICAL and PHYSICS QUANTITIES:
SECONDS_PER_SIDEREAL_DAY = 86164.0905


# MISCELLANEOUS constants and conventions:
ISO_8601_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


# SPECIAL CHARACTERS (unicode), including for astronomy:
DEGREE_SIGN = '\u00B0'
PLUS_MINUS = '\u00B1'
MICRO_SIGN = '\u00B5'
MOON_CHARACTER = '\u263D'  # open to left. Open to right = '\u263E'
# MOON_CHARACTER = '\U0001F319'  # Drat, matplotlib complains 'glyph missing from current font'.
COMET_CHARACTER = '\u2604'
STAR_CHARACTER_WHITE = '\u2606'
STAR_CHARACTER_BLACK = '\u2605'
SUN_CHARACTER_BLACK = '\u2600'
CLOUD_CHARACTER_BLACK = '\u2601'
BULLET_POINT = '\u2022'
INFINITY_SIGN = '\u221E'


