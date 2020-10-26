__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

""" Module 'astrosupport.util'.
    Numerous astro utilities. Bottom of the dependency stack for all EVD's astro software.
    Forked 2020-10-23 from own photrix.util.py.
    Intentions: (1) a separate, importable module for use by all EVD astro python projects.
                (2) freely forkable & useful to the astro python global community. 
    See test file test/test_util.py for comprehensive usage.
"""

# Python core packages:
from datetime import datetime, timedelta, timezone
import math

# External packages:
import ephem  # TODO: remove ephem dependency, in favor of astroplan.




_____CLASSES________________________________________________ = 0


class Timespan:
    """ Holds one (start, end) span of time. Immutable.
        Input: 2 python datetimes (in UTC), defining start and end of timespan.
        methods:
        ts2 = ts.copy()  # (not too useful, as Timespan objects are immutable.)
        ts2 == ts  # only if both start and end are equal
        ts2 = ts.delay_seconds(120)  # returns new Timespan object, offset in both start and end.
        ts.intersect(other)  # returns True iff any overlap at all with other Timespan object.
        ts2 = ts.subtract(other)  # returns new Timespan; longer of 2 possible spans if ambiguous.
        ts.contains_time(t)  # returns True iff ts.start <= t <= ts.end for some datetime object t.
        ts.contains_timespan(other)  # returns True iff ts wholly contains other Timespan.
        Timespan.longer(ts1, ts2)  # returns longer (in duration) of two Timespan objects.
        dt_list = ts.generate_events(jd_ref, period_days, max_events): generates list of up to
            max_events datetimes, all within the Timespan object ts, and the times beginning
            with JD_ref and spaced by period_days.
        str(ts)  # returns string describing Timespan's start, end, and duration in seconds.
    """
    def __init__(self, start_utc, end_utc):
        self.start = start_utc
        self.end = max(start_utc, end_utc)
        self.seconds = (self.end-self.start).seconds
        self.midpoint = self.start + timedelta(seconds=self.seconds / 2)

    def copy(self):
        return Timespan(self.start, self.end)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def delay_seconds(self, seconds):
        delay = timedelta(seconds=seconds)
        return Timespan(self.start+delay, self.end+delay)

    def expand_seconds(self, seconds):
        # Use negative seconds to contract Timespan. New Timespan will have non-negative duration.
        expansion = timedelta(seconds=seconds)
        new_start = min(self.start - expansion, self.midpoint)
        new_end = max(self.end + expansion, self.midpoint)
        return Timespan(new_start, new_end)

    def intersect(self, other):
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        return Timespan(new_start, new_end)

    def subtract(self, other):
        if self.intersect(other).seconds == 0:  # case: no overlap/intersection.
            return self
        if other.contains_timespan(self):  # case: self entirely subtracted away.
            return Timespan(self.start, self.start)
        if self.contains_timespan(other):  # case: 2 timespans -> take the longer.
            diff_early = Timespan(self.start, other.start)
            diff_late = Timespan(other.end, self.end)
            if diff_early.seconds >= diff_late.seconds:
                return diff_early
            else:
                return diff_late
        if self.start < other.start:  # remaining case: partial overlap.
            return Timespan(self.start, other.start)
        else:
            return Timespan(other.end, self.end)

    def contains_time(self, time_utc):
        return self.start <= time_utc <= self.end

    def contains_timespan(self, other):
        return (self.start <= other.start) & (self.end >= other.end)

    @staticmethod
    def longer(ts1, ts2, on_tie="earlier"):
        """
        Returns Timespan with longer duration (larger .seconds).
        If equal duration:
            if_tie=="earlier", return earlier.
            if_tie=="first", return ts1.
            [TODO: add "random" option later to return randomly chosen ts1 or ts2.]
        :param ts1: input Timespan object.
        :param ts2: input Timespan object.
        :param on_tie: "earlier" or "first". Any other string behaves as "first".
        :return: the Timespan object with longer duration.
        """
        if ts1.seconds > ts2.seconds:
            return ts1
        if ts2.seconds > ts1.seconds:
            return ts2
        # here: equal length cases. First, try to break duration tie with earlier midpoint.
        if on_tie.lower() == "earlier" and ts1.midpoint != ts2.midpoint:
            if ts1.midpoint < ts2.midpoint:
                return ts1
            return ts2
        # here, tie-breaking has failed. So simply return first of 2 input Timespans.
        return ts1

    def generate_events(self, jd_ref, period_days, max_events=10):
        """ Returns a list of UTC times of period events within a given Timespan.
        :param jd_ref: Julian Date of any occurence of the period event (e.g., Mira max) [float]
        :param period_days: in days [float]
        :param max_events: maximum number of events to return in list. [int]
        :return: list of up to 10 UTCs of periodic events within the target timespan [list of datetimes]
           Return None if jd_reference or period are invalid. Return empty list if no such events.
        Example: ts.generate_events(2549146.2544, 0.337) will generate UTC datetimes for 2549146.2544,
                     2549146.2544+0.337, 2549146.2544+2*0.337, ... up to 10 datetimes or ts.end, whichever
                     gives the fewer.
        """
        if jd_ref is None or period_days is None:
            return None
        if period_days <= 0.0:
            return None
        jd_ts_start = jd_from_datetime_utc(self.start)
        jd_ts_end = jd_from_datetime_utc(self.end)
        n_prior = math.floor((jd_ts_start - jd_ref) / period_days)
        jd_prior = jd_ref + n_prior * period_days
        utc_list = []
        for i in range(max_events):
            jd_test = jd_prior + i * period_days
            if jd_test > jd_ts_end:
                return utc_list
            if jd_test >= jd_ts_start:
                utc_list.append(datetime_utc_from_jd(jd_test))
        return utc_list

    def __str__(self):
        return "Timespan '" + str(self.start) + "' to '" + str(self.end) + "' = " + \
               str(self.seconds) + " seconds."


class RaDec:
    """ Holds one Right Ascension, Declination sky position (internally as degrees).
        TESTS OK 2020-10-24. """
    def __init__(self, ra, dec):
        """
        :param ra: Right Ascension in degrees [float] or hex format [string].
        :param dec: Declination in degrees [float] or hex format [string].
        :return:
        """
        if isinstance(ra, str):
            self.ra = ra_as_degrees(ra)
        else:
            self.ra = ra
        if isinstance(dec, str):
            self.dec = dec_as_degrees(dec)
        else:
            self.dec = dec
        self.as_degrees = self.ra, self.dec  # stored internally as degrees
        self.as_hex = ra_as_hours(self.ra), dec_as_hex(self.dec)

    def degrees_from(self, other):
        """ Returns great-circle distance of other RaDec object from this one, in degrees.
        :param other: another RaDec object.
        :return: great-circle distance in degrees. [float]
        """
        deg_per_radian = 180.0 / math.pi
        diff_ra = abs(self.ra - other.ra) / deg_per_radian
        cos_dec_1 = math.cos(self.dec / deg_per_radian)
        cos_dec_2 = math.cos(other.dec / deg_per_radian)
        diff_dec = abs(self.dec - other.dec) / deg_per_radian
        arg = math.sqrt(math.sin(diff_dec/2.0)**2 + cos_dec_1*cos_dec_2*math.sin(diff_ra/2.0)**2)
        if arg > 0.001:
            return deg_per_radian * (2.0 * math.asin(arg))  # haversine formula
        else:
            # spherical law of cosines
            sin_dec_1 = math.sin(self.dec / deg_per_radian)
            sin_dec_2 = math.sin(other.dec / deg_per_radian)
            return deg_per_radian * \
                math.acos(sin_dec_1*sin_dec_2 + cos_dec_1*cos_dec_2*math.cos(diff_ra))

    def farther_from(self, other_ra_dec, degrees_limit):
        """ Returns True iff other RaDec object is farther from this one by more than limit.
            Useful for moon avoidance, slew distance, etc.
        :param other_ra_dec: another RaDec object.
        :param degrees_limit: distance limit in degrees. [float]
        :return: True iff distance is greater than given limit, else False. [boolean]
        """
        return self.degrees_from(other_ra_dec) > degrees_limit

    def __eq__(self, other):
        """ True iff other RaDec object is same position as this one. """
        return (self.ra == other.ra) and (self.dec == other.dec)

    def __str__(self):
        """ String representing RaDec object. """
        ra_hex, dec_hex = self.as_hex
        return "RaDec object:  " + ra_hex + "  " + dec_hex

    def __repr__(self):
        """ String representing how RaDec object could have been constructed. """
        ra_hex, dec_hex = self.as_hex
        return "RaDec('" + ra_hex + "', '" + dec_hex + "')"


_____RA_and_DEC_FUNCTIONS_____________________________________ = 0


def ra_as_degrees(ra_string):
    """  Takes Right Ascension as string, returns degrees. TESTS OK 2020-10-24.
    :param ra_string: Right Ascension in either full hex ("12:34:56.7777" or "12 34 56.7777"),
               or degrees ("234.55") [string]
    :return: Right Ascension in degrees between 0 and 360. [float]
    Usage: ra_as_degrees('180.23')    # as degrees from 0 through 360.
           ra_as_degrees('11:16:30')  # as hex, from 0 hours through 24 hours.
    """
    ra_list = parse_hex(ra_string)
    if len(ra_list) == 1:
        ra_degrees = float(ra_list[0])  # input assumed to be in degrees.
    elif len(ra_list) == 2:
        ra_degrees = 15 * (float(ra_list[0]) + float(ra_list[1])/60.0)  # input assumed in hex.
    else:
        ra_degrees = 15 * (float(ra_list[0]) + float(ra_list[1]) / 60.0 +
                           float(ra_list[2])/3600.0)  # input assumed in hex.
    if (ra_degrees < 0) | (ra_degrees > 360):
        ra_degrees = None
    return ra_degrees


def hex_degrees_as_degrees(hex_degrees_string):
    """ Takes angle in hex degrees string (general case) or degrees, returns degrees as float.
        TESTS OK 2020-10-24.
    :param hex_degrees_string: angle in either full hex ("-12:34:56.7777", or "-12 34 56.7777"),
           or degrees ("-24.55")
    :return degrees. [float]
    """
    # dec_list = hex_degrees_string.split(":")
    dec_list = parse_hex(hex_degrees_string)
    # dec_list = [dec.strip() for dec in dec_list]
    if dec_list[0].startswith("-"):
        sign = -1
    else:
        sign = 1
    if len(dec_list) == 1:
        dec_degrees = float(dec_list[0])  # input assumed to be in degrees.
    elif len(dec_list) == 2:
        dec_degrees = sign * (abs(float(dec_list[0])) + float(dec_list[1])/60.0)  # input is hex.
    else:
        dec_degrees = sign * (abs(float(dec_list[0])) + float(dec_list[1]) / 60.0 +
                              float(dec_list[2])/3600.0)  # input is hex.
    return dec_degrees


def dec_as_degrees(dec_string):
    """ Takes Declination as string (hex or degrees), returns degrees as float. TESTS OK 2020-10-24.
    :param dec_string: declination in full hex ("-12:34:56.7777") or degrees ("-24.55"). [string]
    :return: degrees, limited to -90 to +90. [float, or None if outside Dec range]
    """
    dec_degrees = hex_degrees_as_degrees(dec_string)
    if (dec_degrees < -90) | (dec_degrees > +90):
        dec_degrees = None
    return dec_degrees


def ra_as_hours(ra_degrees, seconds_decimal_places=2):
    """ Takes Right Ascension degrees as float, returns RA string. TESTS OK 2020-10-24.
    :param ra_degrees: Right Ascension in degrees, limited to 0 through 360. [float]
    :param seconds_decimal_places: number of places at end of RA string (no period if zero). [int]
    :return: RA in hours/hex format. [string, or None if outside RA range]
    """
    # TODO: Make the decimal places happen (as in degrees_as_hex(), below).
    if (ra_degrees < 0) | (ra_degrees > 360):
        return None
    seconds_decimal_places = int(max(0, seconds_decimal_places))  # ensure int and non-negative.
    total_ra_seconds = ra_degrees * (3600 / 15)
    int_hours = int(total_ra_seconds // 3600)
    remaining_seconds = total_ra_seconds - 3600 * int_hours
    int_minutes = int(remaining_seconds // 60)
    remaining_seconds -= 60 * int_minutes
    if seconds_decimal_places > 0:
        seconds, fract_seconds = divmod(remaining_seconds, 1)
        int_fract_seconds = int(round(fract_seconds * 10 ** seconds_decimal_places))
    else:
        seconds, fract_seconds, int_fract_seconds = round(remaining_seconds), 0, 0
    int_seconds = int(seconds)
    if seconds_decimal_places > 0:
        if int_fract_seconds >= 10 ** seconds_decimal_places:
            int_fract_seconds -= 10 ** seconds_decimal_places
            int_seconds += 1
    if int_seconds >= 60:
        int_seconds -= 60
        int_minutes += 1
    if int_minutes >= 60:
        int_minutes -= 60
        int_hours += 1
    if int_hours >= 24:
        int_hours -= 24
    if seconds_decimal_places > 0:
        format_string = '{0:02d}:{1:02d}:{2:02d}.{3:0' + str(int(seconds_decimal_places)) + 'd}'
    else:
        format_string = '{0:02d}:{1:02d}:{2:02d}'
    ra_string = format_string.format(int_hours, int_minutes, int_seconds, int_fract_seconds)
    return ra_string


def dec_as_hex(dec_degrees, arcseconds_decimal_places=0):
    """ Input: float of Declination in degrees. TESTS OK 2020-10-24.
        Returns: Declination in hex, to desired precision. [string]
    """
    if (dec_degrees < -90) | (dec_degrees > +90):
        return None
    dec_string = degrees_as_hex(dec_degrees, arcseconds_decimal_places)
    return dec_string


def degrees_as_hex(angle_degrees, arcseconds_decimal_places=2):
    """ Takes degrees, returns hex representation. TESTS OK 2020-10-24.
    :param angle_degrees: any angle as degrees. [float]
    :param arcseconds_decimal_places: dec. places at end of hex string (no period if zero). [int]
    :return: same angle in hex notation, with proper sign, unbounded. [string]
    """
    if angle_degrees < 0:
        sign = "-"
    else:
        sign = "+"
    abs_degrees = abs(angle_degrees)
    arcseconds_decimal_places = int(max(0, arcseconds_decimal_places))  # ensure int and non-negative.
    total_arcseconds = abs_degrees * 3600
    int_degrees = int(total_arcseconds // 3600)
    remaining_arcseconds = total_arcseconds - 3600 * int_degrees
    int_arcminutes = int(remaining_arcseconds // 60)
    remaining_arcseconds -= 60 * int_arcminutes
    if arcseconds_decimal_places > 0:
        arcseconds, fract_arcseconds = divmod(remaining_arcseconds, 1)
        int_fract_arcseconds = int(round(fract_arcseconds * 10 ** arcseconds_decimal_places))
    else:
        arcseconds, fract_arcseconds, int_fract_arcseconds = round(remaining_arcseconds), 0, 0
    int_arcseconds = int(arcseconds)
    if arcseconds_decimal_places > 0:
        if int_fract_arcseconds >= 10 ** arcseconds_decimal_places:
            int_fract_arcseconds -= 10 ** arcseconds_decimal_places
            int_arcseconds += 1
    if int_arcseconds >= 60:
        int_arcseconds -= 60
        int_arcminutes += 1
    if int_arcminutes >= 60:
        int_arcminutes -= 60
        int_degrees += 1
    if int_degrees >= 360:
        int_degrees -= 360
    if arcseconds_decimal_places > 0:
        format_string = '{0}{1:02d}:{2:02d}:{3:02d}.{4:0' + str(int(arcseconds_decimal_places)) + 'd}'
    else:
        format_string = '{0}{1:02d}:{2:02d}:{3:02d}'
    hex_string = format_string.format(sign, int(int_degrees), int(int_arcminutes), int_arcseconds,
                                      int_fract_arcseconds)
    return hex_string


_____TIME_and_DATE_FUNCTIONS________________________________ = 0


def get_phase(jd, jd_phase_zero, period_days):
    """ For any Julian Date, Julian Date corresponding to phase zero, and period in days,
        return the phase [float].  TESTS OK 2020-10-24."""
    phase = math.modf((jd - jd_phase_zero) / period_days)[0]
    if phase < 0:
        phase += 1
    return phase


def jd_from_datetime_utc(datetime_utc=None):
    """ For any datetime (UTC), return equivalent Julian Date [float]. TESTS OK 2020-10-24. """
    if datetime_utc is None:
        return None
    datetime_j2000 = datetime(2000, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc)
    jd_j2000 = 2451544.5
    seconds_since_j2000 = (datetime_utc - datetime_j2000).total_seconds()
    return jd_j2000 + seconds_since_j2000 / (24*3600)


def datetime_utc_from_jd(jd=None):
    """ For any Julian Date [float], return datetime UTC.  TESTS OK 2020-10-24. """
    if jd is None:
        return datetime.now(timezone.utc)
    datetime_j2000 = datetime(2000, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc)
    jd_j2000 = 2451544.5
    seconds_since_j2000 = 24 * 3600 * (jd - jd_j2000)
    return datetime_j2000 + timedelta(seconds=seconds_since_j2000)


def hhmm_from_datetime_utc(datetime_utc):
    """ For any datetime (UTC), return string 'hhmm' for the UTC time. TESTS OK 2020-10-24. """
    minutes = round(datetime_utc.hour*60  # NB: banker's rounding (nearest even)
                    + datetime_utc.minute
                    + datetime_utc.second/60
                    + datetime_utc.microsecond/(60*1000000)) % 1440
    hh = minutes // 60
    mm = minutes % 60
    return '{0:0>4d}'.format(100 * hh + mm)


def az_alt_at_datetime_utc(longitude, latitude, target_radec, datetime_utc):
    """ For any earth position, sky position, and datetime (UTC), return the sky position's
        azimuth and altitude in degrees. [2-tuple of floats]  TESTS OK 2020-10-24."""
    # TODO: Migrate this fn from ephem package to something else -- astroplan?
    obs = ephem.Observer()  # for local use.
    if isinstance(longitude, str):
        obs.lon = longitude
    else:
        # next line wrong?: if string should be in deg not radians?? (masked by long passed as hex string?)
        obs.lon = str(longitude * math.pi / 180)
    if isinstance(latitude, str):
        obs.lat = latitude
    else:
        # next line wrong?: if string should be in deg not radians?? (masked by long passed as hex string?)
        obs.lat = str(latitude * math.pi / 180)
    obs.date = datetime_utc
    target_ephem = ephem.FixedBody()  # so named to suggest restricting its use to ephem.
    target_ephem._epoch = '2000'
    target_ephem._ra, target_ephem._dec = target_radec.as_hex  # text: RA in hours, Dec in deg
    target_ephem.compute(obs)
    return target_ephem.az * 180 / math.pi, target_ephem.alt * 180 / math.pi


_____OTHER_UTILITY_FUNCTIONS________________________________ = 0


DEFAULT_LADDER = (1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.4, 8.0, 10.0)  # for ladder_round().


def isfloat(string):
    """ Returns True iff string represents a float number, else False.  TESTS OK 2020-10-24. """
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


def float_or_none(string):
    """ Returns float number iff string represents one, else return None.  TESTS OK 2020-10-24. """
    try:
        return float(string)
    except (ValueError, TypeError):
        return None


def ladder_round(raw_value, ladder=DEFAULT_LADDER, direction="nearest"):
    """
    Rounds to a near-log scale value. May be useful for familiar exposure times.
    Can handle negative numbers, too. Zero returns zero. TESTS OK 2020-10-24.
    :param raw_value: the value we want to round
    :param ladder: ascending list of values from 1 to 10 to which to round.
    :param direction: "nearest" or "down" or "up"
    :return: raw_valued rounded to nearest ladder value, not counting powers of 10,
    e.g., 32.5 -> 32, 111 -> 100, 6321 -> 6400, -126 -> -125
    """
    if raw_value == 0:
        return 0
    base = math.copysign(10**(math.floor(math.log10(math.fabs(raw_value)))), raw_value)
    target = math.fabs(raw_value / base)
    if target in ladder:
        return raw_value
    for i, val in enumerate(ladder[1:]):
        if target < val:
            ratio_below = target / ladder[i]
            ratio_above = ladder[i+1] / target
            if direction == "down":
                return base * ladder[i]
            if direction == "up":
                return base * ladder[i+1]
            if ratio_below <= ratio_above:  # default case "nearest"
                return base * ladder[i]  # round downward
            else:
                return base * ladder[i+1]  # round upward


_____HELPER_FUNCTIONS_______________________________________ = 0


def parse_hex(hex_string):
    """ Helper function for RA and Dec parsing, takes hex string, returns list of floats.
        Not normally called directly by user. TESTS OK 2020-10-24.
    :param hex_string: string in either full hex ("12:34:56.7777" or "12 34 56.7777"),
               or degrees ("234.55")
    :return: list of strings representing floats (hours:min:sec or deg:arcmin:arcsec).
    """
    colon_list = hex_string.split(':')
    space_list = hex_string.split()  # multiple spaces act as one delimiter
    if len(colon_list) >= len(space_list):
        return [x.strip() for x in colon_list]
    return space_list
