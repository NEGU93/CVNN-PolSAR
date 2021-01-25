from pathlib import Path
import re
import numpy as np
from pdb import set_trace
import os

root_path = Path('/media/barrachina/data/datasets/MSTAR/')
stm = {     # State machine for reading file. In order!
    'init': {'next': 'reading_header'},
    'reading_header': {'next': 'reading_data'},
    'reading_data': {'next': 'end'},
    'end': {'next': None}
}


def get_data_files_list():
    data_files = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if re.match(r'.*.[0-9]{3}$', name):     # Regex matching an extension of 3 digits
                data_files.append(os.path.join(path, name))
    return data_files


def read_file(file):
    state = 'init'  # Reset state machine
    metadata = {}
    with open(file, "rb") as f:  # Open as binary
        for line in f:
            if state == 'init':  # Ready to learn
                if b'PhoenixHeaderVer' in line:
                    state = stm[state]['next']
            elif state == 'reading_header':
                if b'EndofPhoenixHeader' in line:
                    state = stm[state]['next']
                    data = np.fromfile(f, dtype='>f4')
                else:
                    key, value = line.decode('utf-8').split('= ')   # Convert it to string and separate key and value
                    metadata[key] = value.rstrip("\n")              # Remove newline from value
            elif state == 'reading_data':
                print("Never get here")
                # data = np.fromfile(f, dtype='>f4')
    if state != 'reading_data':
        print(f"Error: Incomplete data on file: {file}")
    return data, metadata


def get_dataset():
    data_files = get_data_files_list()
    total_files = len(data_files)
    metadata = [{} for _ in range(total_files)]
    data = []
    for file_idx, file in enumerate(data_files):
        dat, meta = read_file(file)
        metadata[file_idx] = meta
        data.append(dat)


if __name__ == '__main__':
    get_dataset()
