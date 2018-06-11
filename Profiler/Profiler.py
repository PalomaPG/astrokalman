import pstats
import sys


class Profiler(object):

    def __init__(self, profile_path):
        p = pstats.Stats(profile_path)
        p.strip_dirs().sort_stats(-1).print_stats()
        p.sort_stats('cumulative').print_stats(10)
        p.sort_stats('time').print_stats(30)
        p.sort_stats('time', 'cumulative').print_stats(.5, 'init')


if __name__ == '__main__':

    Profiler(sys.argv[1])