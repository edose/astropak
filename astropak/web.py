__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core:
from collections import OrderedDict
import datetime
from math import ceil

# External packages:
import requests
from bs4 import BeautifulSoup
import pandas as pd

# From other modules, this package:
from astropak.util import degrees_as_hex, ra_as_degrees, dec_as_degrees, float_or_none


MAX_IDS_PER_MPES_PAGE = 100
MPES_URL_STUB = 'https://cgi.minorplanetcenter.net/cgi-bin/mpeph2.cgi'
LCDB_URL_STUB = 'http://www.minorplanet.info/PHP/generateOneAsteroidInfo.php'
GET_HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) Gecko/20100101 Firefox/64.0'}
MIN_MPES_TABLE_WORDS = 25  # min. white-space-delimited words that can make an ephem table line.
MIN_MP_ALTITUDE = 30
MAX_SUN_ALTITUDE = -9
MAX_V_MAG = 16.5
MIN_MOON_DIST = 40  # degrees

PAYLOAD_DICT_TEMPLATE = OrderedDict([
    ('ty', 'e'),  # e = 'Return Ephemerides'
    ('TextArea', ''),  # the MP IDs
    ('d', '2018+11+22'),  # first utc date
    ('l', '28'),  # number of dates/times (str of integer)
    ('i', '30'),  # interval between ephemerides (str of integer)
    ('u', 'm'),  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    ('uto', '0'),  # UTC offset in hours if u=d
    ('c', ''),   # observatory code
    ('long', '-107.55'),  # longitude in deg; make plus sign safe in code below
    ('lat', '+35.45'),  # latitude in deg; make plus sign safe in code below
    ('alt', '2200'),  # elevation (MPC "altitude") in m
    ('raty', 'a'),  # 'a' = full sexigesimal, 'x' for decimal degrees
    ('s', 't'),  # N/A (total motion and direction)
    ('m', 'm'),  # N/A (motion in arcsec/minute)
    ('igd', 'y'),  # 'y' = suppress line if sun up
    ('ibh', 'y'),  # 'y' = suppress line if MP down
    ('adir', 'S'),  # N/A
    ('oed', ''),  # N/A (display elements)
    ('e', '-2'),  # N/A (no elements output)
    ('resoc', ''),  # N/A (residual blocks)
    ('tit', ''),  # N/A (HTML title)
    ('bu', ''),  # N/A
    ('ch', 'c'),  # N/A
    ('ce', 'f'),  # N/A
    ('js', 'f')  # N/A
])

DF_MPES_COLUMN_ORDER = ['Number', 'Name', 'Code', 'H', 'V_mag',
                        'UTC', 'UTC_text',
                        'RA', 'Dec', 'Phase_angle',
                        'Az', 'Alt',
                        'Motion', 'Motion_pa',
                        'Sun_alt', 'Moon_phase', 'Moon_dist', 'Moon_alt',
                        'Astrometry']


_____MPES_DOWNLOADING_______________________________________ = 0

""" df_mpes columns (as of 2020-11-20):
    [index]: sequential integers from 0. [int]
    Number: MP number. [int]
    Name: MP name. [string]
    Code: MPC coded MP ID. [string]
    H: H-G magnitude. [float]
    V_mag: expected V magnitude. [string, not float]
    UTC: datetime UTC. [pandas timestamp = datetime]
    UTC_text: UTC text, e.g., '2020 11 03 01000'. [string]
    RA: RA as hex text, e.g., '22:00:33.8'. [string]
    Dec: Dec as hex text, e.g., '-12:56:10'. [string]
    Phase_angle: sun-MP-earth phase angle. [float]
    Az: azimuth eastward from north, in degrees. [float]
    Alt: altitude above horizon, in degrees. [int]
    Motion: sky velocity, in arcseconds per minute. [float]
    Motion_pa: sky motion direction, in degrees. [float]
    Sun_alt: altitude of sun, in degrees. [float]
    Moon_phase: phase of moon, 'fullness', 0 to 1, where 1=full, 0=new. [float]
    Moon_dist: MP sky distance from moon, in degrees. [float]
    Moon_alt: moon altitude above horizon, in degrees. [int]
    Astrometry: NA.    
"""


def make_df_mpes_block(mp_start=None, mp_end=None, site_dict=None,
                       utc_start=None, hours=13, file_fullpath=None):
    """ Make dataframe from MPC (MPES) web pages, for a continguous series of MP numbers.
    :param mp_start: lowest of sequential MP numbers to retrieve. [int]
    :param mp_end: highest of sequential MP numbers to retrieve. [int]
    :param site_dict: site-definition dict (usually from make_site_dict()). [python dict]
    :param utc_start: datetime in UTC. [datetime object]
    :param hours: number of hours to retrieve. [int]
    :param file_fullpath: fullpath of file from which to retrieve MPES-equivalent text, rather than
        getting from web. Used for testing. [string]
    """
    mp_list = [i for i in range(mp_start, mp_end + 1)]  # inclusive of start and end MP numbers.
    return make_df_mpes(mp_list, site_dict, utc_start, hours, file_fullpath)


def make_df_mpes(mp_list=None, site_dict=None, utc_start=None, hours=13, file_fullpath=None):
    """ Make dataframe from MPC (MPES, Minor Planet Ephemeris Service) web pages, for a list of MPs
    :param mp_list: list of MP IDs to retrieve, either as integers or strings. [list of strs or ints]
    :param site_dict: site-definition dict (usually from make_site_dict()). [python dict]
    :param utc_start: datetime in UTC. [datetime object]
    :param hours: number of hours to retrieve. [int]
    :param file_fullpath: fullpath of file from which to retrieve MPES-equivalent text, rather than
        getting from web. Used for testing. [string]
    """
    n = len(mp_list)
    if n <= 0:
        return None
    n_calls = ceil(n / MAX_IDS_PER_MPES_PAGE)
    df_mpes = None  # keep IDE happy.
    for i in range(n_calls):
        i_min = i * MAX_IDS_PER_MPES_PAGE
        i_max = min(i_min + MAX_IDS_PER_MPES_PAGE, n)
        mp_sublist = mp_list[i_min: i_max]
        df_sub = make_df_mpes_one_page(mp_sublist, site_dict, utc_start, hours, file_fullpath)
        if i == 0:
            df_mpes = df_sub.copy()
            print('START with', str(mp_list[i_min]), str(mp_list[i_max - 1]))
        else:
            df_mpes = df_mpes.append(df_sub)
            print('         &', str(mp_list[i_min]), str(mp_list[i_max - 1]))
    print('Finished.')
    df_mpes.index = [i for i in range(len(df_mpes))]  # index is sequential integers, from zero.
    return df_mpes


def make_df_mpes_one_page(mp_list=None, site_dict=None, utc_start=None, hours=13, file_fullpath=None):
    """ Get MPC (MPES, Minor Planet Ephemeris Service) text for a list of MPs, for a given location and
        date, return dataframe of relevant data parsed from that page.  
        Adapted from mp_phot::mp_astrometry.py::get_one_html_from_text() and ::html().
    :param mp_list: list of MP IDs to retrieve, either as integers or strings. [list of strs or ints]
    :param site_dict: site-definition dict (usually from make_site_dict()). [python dict]
    :param utc_start: datetime in UTC. [datetime object]
    :param hours: number of hours to retrieve. [int]
    :param file_fullpath: fullpath of file from which to retrieve MPES-equivalent text, rather than
        getting from web. Used for testing. [string]
    :return: 
    """
    if file_fullpath is not None:
        with open(file_fullpath, 'r') as f:
            lines = f.readlines()
    else:
        if len(mp_list) > MAX_IDS_PER_MPES_PAGE:
            print(' >>>>> WARNING:', str(len(mp_list)), 'items requested,', str(MAX_IDS_PER_MPES_PAGE),
                  'is the maximum. Using first', str(MAX_IDS_PER_MPES_PAGE), ' only.')
            mp_list = mp_list[:MAX_IDS_PER_MPES_PAGE]

        payload_dict = PAYLOAD_DICT_TEMPLATE
        payload_dict['TextArea'] = '%0D%0A'.join([str(mp) for mp in mp_list])
        payload_dict['d'] = utc_as_mpes_decimal(utc_start)
        payload_dict['l'] = str(int(round(hours)))
        payload_dict['i'] = '1'
        payload_dict['u'] = 'h'
        payload_dict['c'] = site_dict['mpc code']  # MPC code required to get 'Observations Needed' status.
        if site_dict['mpc code'] is None:
            site_dict['mpc code'] = ''
        if site_dict['mpc code'].strip() != '':
            payload_dict['long'] = ''
            payload_dict['lat'] = ''
            payload_dict['alt'] = ''
        else:
            payload_dict['long'] = str(site_dict['longitude'])
            payload_dict['lat'] = str(site_dict['latitude'])
            payload_dict['alt'] = str(site_dict['elevation'])

        # Construct URL and header (get), get html as text lines:
        payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])        
        url = MPES_URL_STUB + '/?' + payload_string
        r = requests.get(url)
        lines = r.text.splitlines()
        # #################### ONE-TIME RUN for making test file only. DELETE WHEN DONE.
        # with open('C:/Dev/astropak/test/$data_for_test/MPES_page_example.txt', 'w') as f:
        #     f.writelines('\n'.join(lines))
        # #################### END TEST BLOCK. DELETE WHEN DONE.
    
    mp_blocks = chop_into_mp_blocks(lines)
    df_mpes_one_page = make_df_from_mp_blocks(mp_blocks)
    return df_mpes_one_page


def utc_as_mpes_decimal(utc):
    """ Take datetime in UTC, return string like 2020 11 01.15 suitable for MPES use.    
    :param utc: datetime in UTC. [datetime object]
    :return: 'mm+dd' representation of month and day. [string]
    """
    month_string = '{0:04d}+{1:02d}'.format(utc.year, utc.month)
    decimal = utc.hour / 24 + utc.minute / 24 / 60 + utc.second / 24 / 3600
    if utc.minute == 0 and utc.second == 0:
        day_string = '{0:02d}'.format(utc.day)
    else:
        decimal = utc.hour / 24 + utc.minute / 24 / 60 + utc.second / 24 / 3600
        day_string = '{0:05.2f}'.format(decimal)
    return month_string + '+' + day_string


def chop_into_mp_blocks(html_lines):
    """ Take lines of MPES html page, return list of sublists of strings, each sublist for one MP.
    Adapted from mp_phot::mp_astrometry.py::chop_html().
    :param html_lines: all lines of one MPES page. [list of strings]
    :return: list of sublists of strings, each sublist for one MP. [list of lists of strings]
    """
    # Collect lines numbers for all vertical block delimiters (including end of file):
    hr_line_numbers = [0]
    for i_line, line in enumerate(html_lines):
        if '<hr>' in line:
            hr_line_numbers.append(i_line)
    hr_line_numbers.append(len(html_lines) - 1)

    # Make a block if MP data actually found between two successive horizontal lines:
    mp_blocks = []
    for i_hr_line in range(len(hr_line_numbers) - 2):
        for i_line in range(hr_line_numbers[i_hr_line], hr_line_numbers[i_hr_line + 1]):
            if html_lines[i_line].strip().lower().startswith('<p>discovery date'):
                mp_block = html_lines[hr_line_numbers[i_hr_line]: hr_line_numbers[i_hr_line + 1] + 1]
                mp_blocks.append(mp_block)
                break
    return mp_blocks


def make_df_from_mp_blocks(mp_blocks):
    """ Take mp_blocks (html) and make a dataframe of the contents.
    :param mp_blocks: list of sublists of strings, one sublist per MP. [list of lists of strings]
    :return: dataframe of results from all mp_blocks. [pandas Dataframe]
    """
    dict_list = []
    for block in mp_blocks:
        lines_as_read = block
        lines = [line.strip() for line in lines_as_read]
        mp_dict = dict()
        mp_dict['Astrometry'] = 'NA'  # default when 'Futher observations?' line is missing (it happens).

        # Handle header lines of mp block:
        for i, line in enumerate(lines):
            if line.startswith('<b>'):
                mp_dict['Number'] = line.split(')')[0].split('(')[-1].strip()
                mp_dict['Name'] = line.split(')')[-1].split('<')[0].strip()
            if line.startswith('Last observed on'):
                mp_dict['last_obs'] = line[len('Last observed on'):].strip().replace('.', '')
            if 'further observations?' in line.lower():
                raw_status = line.split('>', maxsplit=1)[1].strip().lower()
                if raw_status.startswith('not necessary'):
                    mp_dict['Astrometry'] = 'not needed'
                if raw_status.startswith('useful'):
                    mp_dict['Astrometry'] = 'useful'
            if line.startswith('<p><pre>'):
                if i < len(block) - 1:
                    mp_dict['Code'] = lines[i + 1].strip().split()[0]
                    h_str = lines[i+1].split('[H=', maxsplit=1)[1].split(']')[0].strip()
                    mp_dict['H'] = float_or_none(h_str)

        # Handle ephemeris table of mp block:
        table_start_found = False
        for line in block:
            if '<pre>' in line:
                table_start_found = True
                continue  # start of table found.
            if table_start_found and '</pre>' in line:
                break
            line_dict = mp_dict.copy()
            items = line.split()
            if len(items) >= MIN_MPES_TABLE_WORDS:
                mp_alt = float(items[18])
                v_mag = float(items[14])
                sun_alt = float(items[19])
                moon_dist = float(items[21])
                moon_alt = float(items[22])
                high_enough = mp_alt >= MIN_MP_ALTITUDE
                bright_enough = v_mag <= MAX_V_MAG
                sun_low_enough = sun_alt <= MAX_SUN_ALTITUDE
                moon_distant_enough = (moon_dist >= MIN_MOON_DIST or moon_alt < 0)
                if high_enough and bright_enough and sun_low_enough and moon_distant_enough:
                    line_dict['V_mag'] = '{:.1f}'.format(v_mag)                    
                    line_dict['UTC_text'] = ' '.join(items[0:4])
                    year, month, day, hms = tuple(items[0:4])
                    hour = int(hms[0:2])
                    minute = int(hms[2:4])
                    second = int(hms[4:6])
                    line_dict['UTC'] = datetime.datetime(int(year), int(month), int(day),
                                                         hour, minute, second,
                                                         tzinfo=datetime.timezone.utc)
                    line_dict['RA'] = ':'.join(items[4:7])
                    line_dict['Dec'] = ':'.join(items[7:10])
                    line_dict['Phase_angle'] = float(items[13])
                    line_dict['Az'] = float(items[17])
                    line_dict['Alt'] = round(mp_alt)
                    line_dict['Motion'] = float(items[15])
                    line_dict['Motion_pa'] = float(items[16])
                    line_dict['Sun_alt'] = sun_alt
                    line_dict['Moon_phase'] = float(items[20])
                    line_dict['Moon_dist'] = float(items[21])
                    line_dict['Moon_alt'] = round(moon_alt)
                    # ra_degrees = ra_as_degrees(mp_dict['ra'])
                    # dec_degrees = dec_as_degrees(mp_dict['dec'])
                    # date_string = ''.join(items[0:3])
                    dict_list.append(line_dict)
    df = pd.DataFrame(dict_list).reindex(columns=DF_MPES_COLUMN_ORDER)
    return df
                        

_____LCDB_ONE_ASTEROID_DOWNLOADING__________________________ = 0


def try_lcdb(mp_number=4954, site_dict=None, utc_start='2020-11-02', file_fullpath=None):
    """ Download and parse one minorplanet.info 'One Asteroid' page, parse it, save data as dataframe.
        Not yet needed for color-index historical data screening, so suspend development 2020-11-01.
    """
    if file_fullpath is not None:
        with open(file_fullpath, 'r') as f:
            lines = f.readlines()
    else:
        url = 'http://www.minorplanet.info/PHP/generateOneAsteroidInfo.php/'
        longitude = site_dict['longitude'] \
            if site_dict['longitude'] <= 180 \
            else site_dict['longitude'] - 360
        parameter_dict = {'AstNumber': str(mp_number),
                          'AstName': '',
                          'Longitude': str(longitude),
                          'Latitude': str(site_dict['latitude']),
                          'StartDate': utc_start,
                          'UT': '7',
                          'subOneShot': 'Submit'}
        r = requests.post(url, data=parameter_dict)
        soup = BeautifulSoup(r.text, features='html5lib')
        tables = soup.find_all('table')
        ci_table = [t for t in tables if 'SGErr' in t.strings][-1]  # Color index table; OK to here.
        # TODO: Complete this parsing if and when needed.
        return ci_table


_____LCDB_COLOR_INDEX_DATABASE______________________________ = 0

class LCDB_Colors:
    """ Contains data from one LCDB Color Index file (probably named LC_COLORINDEX_PUB.TXT) as downloaded
        from minorplanet.info (their current DB).
    """
    def __init__(self, fullpath):
        self.contents = dict()  # accumulator
        self.is_valid = None    # placeholder
        if fullpath is not None:
            with open(fullpath, 'r') as f:
                lines = f.readlines()
        if not lines[0].startswith('ASTEROID LIGHTCURVE') or not lines[1].startswith('GENERATED: '):
            print(' >>>>> ERROR: File header does not look like a Color Index file.')
            self.is_valid = False
            return
        table_start_found = False
        table_header_found = False
        mp_number, mp_name, flag, period, amplitude = None, None, None, None, None
        color_columns, color_list = [], []
        for i, line in enumerate(lines):
            # Skip blank lines:
            if line.strip() == '':
                continue
            # Skip lines until table start found:
            if not table_start_found:
                if line.startswith(50 * '-'):
                    table_start_found = True
                    continue
            # If table header, parse color columns:
            if not table_header_found:
                if line.startswith('Number   '):
                    table_header_found = True
                    color_start_columns = []
                    pieces = line.rpartition(' Amp ')
                    color_names = pieces[2].split()
                    i_start = len(pieces[0] + pieces[1])
                    for color_name in color_names:
                        i_color_name = line.find(color_name, i_start)
                        i_start += len(color_name)
                        color_start_columns.append(i_color_name)
                    color_columns = []
                    for ii, col in enumerate(color_start_columns):
                        if ii < len(color_start_columns) - 1:
                            color_columns.append((color_names[ii], col, color_start_columns[ii + 1]))
                        else:
                            color_columns.append((color_names[ii], col, len(line) - 1))
                continue
            # Case: line is MP record header:
            if line[:7].strip() != '':
                if mp_number is not None and len(color_list) >= 1:
                    self._save_record(mp_number, mp_name, flag, period, amplitude, color_list)
                    mp_number = None
                mp_number = int(line[:7])
                mp_name = line[10:41].strip()
                flag = line[8]
                period = float_or_none(line[41:55])
                amplitude = float_or_none(line[55:59])
                color_list = []
                continue
            # Case: line containing color-index data:
            if line[:10].strip() == '' and line[10:35].strip() != '':
                for name, start, end in color_columns:
                    color_string = line[start:end]
                    if color_string.strip() != '':
                        color_list.append((name, float(color_string)))
                    iiii = 4
            iiii = 4
        # Ensure last item is saved:
        if mp_number is not None and len(color_list) >= 1:
            self._save_record(mp_number, mp_name, flag, period, amplitude, color_list)

    def _save_record(self, mp_number, mp_name, flag, period, amplitude, color_list):
        key = mp_number if mp_number != 0 else mp_name
        self.contents[key] = {
            'mp_number': mp_number,
            'mp_name': mp_name,
            'flag': flag,
            'period': period,
            'amplitude': amplitude,
            'color_list': color_list
        }

    def color_count(self, mp_id):
        """ Return total number of color values for this MP ID. """
        record = self.contents.get(mp_id, None)
        return 0 if record is None else len(record['color_list'])

    def sloan_count(self, mp_id):
        """ Return number of Sloan color values for this MP ID. """
        record = self.contents.get(mp_id, None)
        if record is None:
            return 0
        sloan_values = [1 for entry in record['color_list'] if entry[0] in ['SGR', 'SRI', 'SIZ']]
        return len(sloan_values)

    def jc_count(self, mp_id):
        """ Return number of Johnson-Cousins color values for this MP ID. """
        record = self.contents.get(mp_id, None)
        if record is None:
            return 0
        jc_values = [1 for entry in record['color_list'] if entry[0] in ['BV', 'BR', 'VR', 'VI', 'RI']]
        return len(jc_values)

    def has_id(self, mp_id):
        """ Returns True iff mp_id has any record in this database file, else returns False. """
        return mp_id in self.contents.keys()



