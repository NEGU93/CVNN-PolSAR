from data_processing import load_matlab_matrices
from pdb import set_trace


def load_gilles_mat_data(fname, default_path="/media/barrachina/data/gilles_data/"):
    mat = load_matlab_matrices(fname, default_path)
    ic = mat['ic'] .squeeze(axis=0)             # Labels corresponding to types
    nb_sig = mat['nb_sig'].squeeze(axis=0)      # number of examples for each label (label=position_ic)
    sx = mat['sx'][0]                           # Unknown scalar
    types = [t[0] for t in mat['types'].squeeze(axis=0)]    # labels legends
    xp = []                                     # Metadata TODO: good for regression network
    for t in mat['xp'].squeeze(axis=1):
        xp.append({'Type': t[0][0], 'Nb_rec': t[1][0][0], 'Amplitude': t[2][0][0], 'f0': t[3][0][0],
                   'Bande': t[4][0][0], 'Retard': t[5][0][0], 'Retard2': t[6][0][0], 'Sequence': t[7][0][0]})

    xx = mat['xx'].squeeze(axis=2).squeeze(axis=1).transpose()      # Signal data

    return ic, nb_sig, sx, types, xp, xx


if __name__ == '__main__':
    ic, nb_sig, sx, types, xp, xx = load_gilles_mat_data("data_cnn1dT.mat")
    set_trace()
