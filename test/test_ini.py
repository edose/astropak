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

    # Test normal case, template in directory just above that of ini file:
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == True
    assert len(i.warnings) == 0
    assert i.value_dict['key the first'] == 11
    assert isinstance(i.value_dict['key the first'], int)
    assert i.value_dict['key 2'] == 34.33
    assert isinstance(i.value_dict['key 2'], float)
    assert i.value_dict['cle troisieme'] == 'this is a string, mate'
    assert i['key 2'] == i.value_dict['key 2']
    assert i['also sprach vier'] == ['this remains a', 'multiline string']
    assert i['to one line'] == 'multiline string, joined to one string only.'
    assert i['how about it'] == False
    assert i['not a key'] is None

    # Test normal case, template in specified directory path:
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1_template_specified_path.ini')
    template_dir_path = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                     'template_below_ini')
    i = ini.IniFile(ini_fullpath, template_directory_path=template_dir_path)
    assert i.is_valid == True
    assert len(i.warnings) == 0
    assert i.value_dict['key the first'] == 11
    assert i.value_dict['ofodiodo'] == 34.335
    assert isinstance(i.value_dict['key the first'], int)

    # Case: ini file not found:
    i = ini.IniFile(ini_fullpath + 'xxx')
    assert i.is_valid == False
    assert i.warnings[0].startswith('Ini file not found')

    # Case: template file not found in ini dir or its parent dir:
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-no_template.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == False
    assert i.warnings[0].startswith('Template path not specified, and template not found in ini dir')

    # Case: template file not found in specified directory path:
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-no_template.ini')
    specified_template_path = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini')
    i = ini.IniFile(ini_fullpath, template_directory_path=specified_template_path)
    assert i.is_valid == False
    assert i.warnings[0].startswith('Template not found at')

    # Case: a key (from template) not found in ini file (not fatal):
    ini_fullpath = os.path.join(TEST_TOP_DIRECTORY, '$data_for_test', 'ini', 'testini',
                                'test1-missing_key.ini')
    i = ini.IniFile(ini_fullpath)
    assert i.is_valid == True
    assert len(i.warnings) >= 1
    assert 'Value for [Section 1][troisieme] not found in .ini file.' in i.warnings
    assert 'Value for [Section 1][how about] not found in .ini file.' in i.warnings
    assert 'Value for [Section 1][to one line] not found in .ini file.' in i.warnings

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
    assert 'Value for [Section 1][key 2] cannot be parsed as float: \'not a float as required\'' \
           in i.warnings
    assert 'Value for [Section 1][how about] not found in .ini file.' in i.warnings
    assert 'Value for [Section 1][to one line] not found in .ini file.' in i.warnings
