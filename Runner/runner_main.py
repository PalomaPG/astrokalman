from Runner import Runner
import sys


def main():

    runner = Runner(sys.argv[1])
    print(runner.get_mjd_lst())


if __name__ == '__main__':
    main()