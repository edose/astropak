__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pytest

# TARGET TEST MODULE:
from astropak import ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test")


def test_class_inifile():
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini', 'test1.ini')

    # Test normal case:
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == True
    assert len(i.warnings) == 0
    assert i.value_dict['key the first'] == 11
    assert isinstance(i.value_dict['key the first'], int)
    assert i.value_dict['key 2'] == 34.33
    assert isinstance(i.value_dict['key 2'], float)
    assert i.value_dict['cle troisieme'] == ['this is a string, mate']
    assert isinstance(i.value_dict['cle troisieme'][0], str)
    assert i['key 2'] == i.value_dict['key 2']
    assert i['not a key'] is None

    # Case: ini file not found:
    i = ini.IniFile(ini_fullpath + 'xxx')
    assert i.is_valid == False
    assert i.warnings[0].startswith('Ini file not found')

    # Case: template filename garbled or template file not found:
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-no_template.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == False
    assert i.warnings[0].startswith('Template file not found')

    # Case: a key (from template) not found in ini file (not fatal):
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-missing_key.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == True
    assert i.warnings[0].startswith('Entry not found in .ini file')

    # Case: invalid value type in template:
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-bad_value_type.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == False
    assert i.warnings[0].startswith('Invalid value type')

    # Case: value type valid but ini value not of that type (requires try/catch):
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-wrong_type.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == False
    assert i.warnings[0].startswith('Cannot be parsed as float')
