from Runner import Runner
import sys


def main():

    runner = Runner(sys.argv[1])
    #print('Lista de archivos PSF')
    print(len(runner.get_files('baseDir')))
    print(len(runner.get_files('maskDir')))
    print(len(runner.get_files('crblastDir')))
    #print(len(runner.get_files('scienceDir')))


if __name__ == '__main__':
    main()