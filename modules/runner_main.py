from DataPicker import DataPicker
import sys


def main():

    picker = DataPicker(sys.argv[1])
    #print('Lista de archivos PSF')
    print(len(picker.get_files('baseDir')))
    print(len(picker.get_files('maskDir')))
    print(len(picker.get_files('crblastDir')))
    #print(len(runner.get_files('scienceDir')))


if __name__ == '__main__':
    main()