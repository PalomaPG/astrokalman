import cProfile
import pstats
import io


class Profiler(object):

    def __init__(self):
        command = ''
        stats_name = ''
