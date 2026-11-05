'''
    Utility for folder
    Coded and annotated by Yuxuan SONG
    Last modified time: 5 Dec 2025
'''
from os.path import exists
from os import makedirs


def generate_folder(folder_path: str):
    r'''Generate folder path if not exists yet.

    Args:
        folder_path (str): Folder path.
    '''

    if exists(folder_path) is False:
        makedirs(folder_path)
        print('Folder \'{:s}\' is created.'.format(folder_path))
