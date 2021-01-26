from pathlib import Path
import re
import numpy as np
from pdb import set_trace
import os
from tqdm import tqdm
from pprint import pprint

root_path = Path('/media/barrachina/data/datasets/MSTAR/')
stm = {     # State machine for reading file. In order!
    'init': {'next': 'reading_header'},
    'reading_header': {'next': 'end'},
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
                if b'EndofPhoenixHeader' in line:   # End of the header reached
                    state = stm[state]['next']      # Next state
                    # All that it's left is to read the data so why not read it now?
                    tmp_data = np.fromfile(f, dtype='>f4')
                    assert len(tmp_data) % 2 == 0, "Error, obtained array had not a pair number of elements."
                    magnitude = tmp_data[:int(len(tmp_data)/2)]
                    direction = np.exp(1j*tmp_data[int(len(tmp_data)/2):])
                    data = np.multiply(magnitude, direction)    # Element-wise multiplication
                    data = np.reshape(data, (metadata['NumberOfRows'], metadata['NumberOfColumns']))
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
    return data, metadata


def get_dataset():
    data_files = get_data_files_list()
    total_files = len(data_files)
    metadata = [{} for _ in range(total_files)]
    data = []
    for file_idx, file in enumerate(tqdm(data_files)):
        path = Path(file).parent.absolute()
        dat, meta = read_file(file)
        metadata[file_idx] = meta
        metadata[file_idx]['path'] = str(path).replace(str(root_path), '')
        data.append(dat)
    assert len(metadata) == len(data), "An error occurred, metadata has different size than data"
    return metadata, data


def save_dataset(metadata, data, path):
    np_data = np.array(data, dtype=object)
    np_metadata = np.array(metadata)
    np.save(path/'data.npy', np_data)
    np.save(path/'metadata.npy', np_metadata)


if __name__ == '__main__':
    metadata, data = get_dataset()
    save_dataset(metadata, data, root_path)

