from modules.RoutineHandler import RoutineHandler
from modules.TPDetector import TPDetector
import sys

if __name__ == '__main__':

    rh = RoutineHandler(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    rh.process_settings()
    #rh.iterate_over_sequences()
    rh.iterate_over_sequences(check_found_objects=True)
    #rh.plot_results()
