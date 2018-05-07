from abc import abstractmethod


class AbstractRunner(object):

    def __init__(self, year, filter='lk'):

        # Files in Leftraru
        self.base_input = '/home/apps/astro/data/ARCHIVE'

        # Output files path
        self.base_output = None

        # Applied filter
        self.filter = filter

    def set_parameters(self):
        pass





