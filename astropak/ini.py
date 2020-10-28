__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
import os.path
import configparser
from collections import OrderedDict


class IniFile:
    """ Holds one .ini file's contents in easily retrievable form (like a dict). Immutable.
        Reads .ini file, stores data, delivers on demand.
        Available value types for template:
            int, float, string : holds value in type indicated. Multiline strings are joined by a space.
            string list: holds string or multiline string as a *list* of strings, even if only one line.
    """
    def __init__(self, inifile_fullpath):
        """ Takes template OrderedDict and .ini file name, creates object.
        :param inifile_fullpath: fullpath of .ini file to read & store. [string]
        """
        # Declare fields (as placeholders):
        self.inifile_fullpath = inifile_fullpath
        self.value_dict = OrderedDict()
        self.warnings = []
        self.is_valid = True  # presumptive value to be negated on error.

        # Read .ini file into ini_config:
        if not(os.path.exists(inifile_fullpath) and os.path.isfile(inifile_fullpath)):
            self.warnings.append('Ini file not found: ' + inifile_fullpath)
            self.is_valid = False
            return
        self.ini_config = configparser.ConfigParser()

        self.ini_config.read(inifile_fullpath)

        # Find and read template for this .ini file:
        # TODO: Next line goes to exception when [Ini Template] absent--should give warning and exit.
        self.template_filename = self.ini_config.get('Ini Template', 'Filename')
        if self.template_filename is None:
            self.warnings.append('Template filename not parsed. See top of .ini file.')
            self.is_valid = False
            return
        self.template_fullpath = self.find_template_file()
        if self.template_fullpath is None:
            self.warnings.append('Template file not found. See top of .ini file.')
            self.is_valid = False
            return
        # Now that we have template fullpath, open the template file:
        template_config = configparser.ConfigParser()
        template_config.read(self.template_fullpath)

        # Loop through all template elements, build value_dict:
        for section in template_config.sections():
            if section.lower().strip() != 'ini template':  # just for safety.
                for key in template_config[section]:
                    template_value_string = template_config.get(section, key)
                    self.parse_and_store_one_entry(section, key, template_value_string)

    def find_template_file(self):
        """ Find template file in likely directories, return fullpath. """
        inifile_path = os.path.dirname(self.inifile_fullpath)
        # First, seek template file in same directory as .ini file itself:
        trial_directory_path = inifile_path
        trial_template_fullpath = os.path.join(trial_directory_path, self.template_filename)
        if os.path.exists(trial_template_fullpath) and os.path.isfile(trial_template_fullpath):
            return trial_template_fullpath

        # Next, seek template in next directory up:
        trial_directory_path = os.path.dirname(trial_directory_path)
        trial_template_fullpath = os.path.join(trial_directory_path, self.template_filename)
        if os.path.exists(trial_template_fullpath) and os.path.isfile(trial_template_fullpath):
            return trial_template_fullpath

        return None  # signals 'template file not found.'

    def parse_and_store_one_entry(self, section, key, template_value_string):
        """ Parse value_string(s) into (1) key and (2) value to store, then add to storage dict. """
        template_items = template_value_string.strip().split('->', maxsplit=1)
        value_type = template_items[0].strip().lower()
        value = self.ini_config.get(section, key, fallback=None)
        # value = self.ini_config[section][key].strip()
        if value is None:
            self.warnings.append('Entry not found in .ini file [' + section + '][' + key + '].')
            return

        if value_type == 'int':
            try:
                value_to_store = int(value)
            except (TypeError, ValueError):
                self.warnings.append('Cannot be parsed as int: ' + value)
                self.is_valid = False
                return
        elif value_type == 'float':
            try:
                value_to_store = float(value)  # TODO: add try/catch.
            except (TypeError, ValueError):
                self.warnings.append('Cannot be parsed as float: ' + value)
                self.is_valid = False
                return
        elif value_type in ['boolean', 'bool']:
            try:
                value_to_store = self.ini_config.getboolean(section, key)
            except (TypeError, ValueError):
                self.warnings.append('Cannot be parsed as boolean: ' + value)
                self.is_valid = False
                return
        elif value_type in ['str', 'string', 'string list']:
            value_to_store = value.splitlines()
            if not value_type == 'string list':
                value_to_store = ' '.join(value_to_store)  # list -> one string
        else:
            self.warnings.append('Invalid value type \'' + value_type + '\' in template[' +
                                 section + '][' + key + '].')
            self.is_valid = False
            return
        ini_key = template_items[1].strip().lower()
        if ini_key in self.value_dict:
            self.warnings.append('Key \'' + ini_key + '\' used twice in template.')
        self.value_dict[ini_key] = value_to_store

    def __getitem__(self, key):
        """ For given item key, return value. """
        return self.value_dict.get(key.lower().strip(), None)
