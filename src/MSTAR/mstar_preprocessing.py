from pathlib import Path
import re
import numpy as np
import pandas as pd
from pdb import set_trace
import os
from tqdm import tqdm
from pprint import pprint

"""
This script is used to read the MSTAR dataset and save it to be easily accesible later
"""

root_path = Path('/media/barrachina/data/datasets/MSTAR/')
stm = {     # State machine for reading file. In order!
    'init': {'next': 'reading_header'},
    'reading_header': {'next': 'end'},
    'end': {'next': None}
}


def get_data_files_list():
    """
    Reads all files inside the root file (and sub-folders) and returns the list of those that has
    3 digit extensions (ex: .015)
    :return: The list of all files with the 3 digit extensions
    """
    data_files = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if re.match(r'.*.[0-9]{3}$', name):     # Regex matching an extension of 3 digits
                data_files.append(os.path.join(path, name))
    return data_files


def read_file(file):
    """
    Reads the file
    :param file: File to read
    :return: Dictionary with all the metadata of the header and the data as the key 'data' of the dictionary.
    """
    state = 'init'  # Reset state machine
    metadata = {}
    with open(file, "rb") as f:  # Open as binary
        for line in f:
            if state == 'init':  # Ready to learn
                if b'PhoenixHeaderVer' in line:
                    state = stm[state]['next']
            elif state == 'reading_header':
                if b'EndofPhoenixHeader' in line:   # End of the header reached
                    state = stm[state]['next']      # Next state
                    # All that it's left is to read the data so why not read it now?
                    tmp_data = np.fromfile(f, dtype='>f4')
                    assert len(tmp_data) % 2 == 0, "Error, obtained array had not a pair number of elements."
                    magnitude = tmp_data[:int(len(tmp_data)/2)]
                    direction = np.exp(1j*tmp_data[int(len(tmp_data)/2):])
                    dat = np.multiply(magnitude, direction)    # Element-wise multiplication
                    dat = np.reshape(dat, (metadata['NumberOfRows'], metadata['NumberOfColumns']))
                    metadata['data'] = dat
                else:
                    key, str_value = line.decode('utf-8').split('= ')  # Convert it to string and separate key and value
                    str_value = str_value.rstrip("\n")
                    try:
                        value = int(str_value)
                    except ValueError:
                        try:
                            value = float(str_value)
                        except ValueError:
                            value = str_value
                    metadata[key] = value              # Remove newline from value
            elif state == 'end':
                raise IOError("np.fromfile didn't read everything")
    if state != 'end':
        raise IOError(f"Error: Incomplete data on file: {file}")
    return metadata


def get_dataset():
    """
    Get all the MSTAR dataset as a list of dictionaries.
    :return: A list of dictionaries with all the data. Each dictionary of the list corresponds to a file.
    """
    data_files = get_data_files_list()
    total_files = len(data_files)
    data = [{} for _ in range(total_files)]
    for file_idx, file in enumerate(tqdm(data_files)):
        path = Path(file).parent.absolute()
        dat = read_file(file)
        data[file_idx] = dat
        data[file_idx]['path'] = str(path).replace(str(root_path), '')
    assert len(data) == len(data), "An error occurred, metadata has different size than data"
    return data


def save_dataset(data, path):
    """
    Saves the dataset, if it is a pandas dataframe it will pickle it,
        if not, it will try to cast to numpy array and save it as .npy.
    :param data: Data to be saved
    :param path: Path to save the data
    :return: None
    """
    if isinstance(data, pd.DataFrame):
        df.to_pickle(path/'data.pkl')
    else:
        np.save(path/'data.npy', np.array(data))


if __name__ == '__main__':
    data = get_dataset()
    df = pd.DataFrame(data)
    save_dataset(df, root_path)
    save_dataset(data, root_path)

