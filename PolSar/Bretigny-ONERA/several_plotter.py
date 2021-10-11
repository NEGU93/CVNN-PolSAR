import plotly.express as px
from skimage.io import imread
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
from pdb import set_trace
from typing import Optional

params = {
    "dtype": 'complex',
    "model": 'default',
    "library": 'cvnn',
    "dataset_mode": 'k',
    "dataset_split": 'random',
    "boxcar": 'default',
    "loss": 'conventional',
}
keys = ['accuracy', 'val_accuracy']


def get_paths(root_dir: str = "/media/barrachina/data/results/Bretigny/after_icassp") -> dict:
    """
    Finds all paths in a given `root_dir` directory
    :param root_dir:
    :return:
    """
    child_dirs = os.walk(root_dir)
    monte_dict = {}
    for child_dir in child_dirs:
        if "run-" in child_dir[0].split('/')[-1]:
            file_path = Path(child_dir[0]) / "model_summary.txt"
            if file_path.is_file():
                with open(file_path) as txt_sum_file:
                    simu_params = txt_sum_file.readline()
                    if (Path(child_dir[0]) / 'history_dict').is_file():
                        try:
                            boxcar_index = simu_params.split().index('--boxcar')
                        except ValueError:
                            boxcar_index = -1
                        try:
                            model_index = simu_params.split().index('--model')
                        except ValueError:
                            model_index = -1
                        monte_dict[child_dir[0]] = {
                            "data": str(Path(child_dir[0]) / 'history_dict'),
                            "image": str(Path(child_dir[0]) / 'prediction.png'),
                            "params": {
                                "dtype": f"{'real' if 'real_mode' in simu_params else 'complex'}",
                                "model": f"{simu_params.split()[model_index + 1] if model_index != -1 else 'default'}",
                                "library": f"{'cvnn' if 'tensorflow' not in simu_params else 'tensorflow'}",
                                "dataset_mode": f"{'coh' if 'coherency' in simu_params else 'k'}",
                                "dataset_split": f"{'sections' if 'split_datasets' in simu_params else 'random'}",
                                "boxcar": f"{simu_params.split()[boxcar_index + 1] if boxcar_index != -1 else 'default'}",
                                "loss": f"{'conventional' if 'weighted_loss' not in simu_params else 'weighted'}",
                            }
                        }
                    else:
                        print("No history_dict found on path " + child_dir[0])
            else:
                print("No model_summary.txt found in " + child_dir[0])
    return monte_dict


user_dict = get_paths("/media/barrachina/data/results/Bretigny/after_icassp")

"""
    Update menus dictionaries
"""


def _get_dataset_mode_dict(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'k',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_mode": "k"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'coh',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_mode": "coh"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_dtype(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'complex',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dtype": "complex"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'real',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dtype": "real"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_library(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'cvnn',
                'args': [
                    {'y': [
                        _update_plot(new_value={"library": "cvnn"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'tensorflow',
                'args': [
                    {'y': [
                        _update_plot(new_value={"library": "tensorflow"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_dataset_split(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'random',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_split": "random"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'sections',
                'args': [
                    {'y': [
                        _update_plot(new_value={"dataset_split": "sections"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_loss(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'conventional',
                'args': [
                    {'y': [
                        _update_plot(new_value={"loss": "conventional"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': 'weighted',
                'args': [
                    {'y': [
                        _update_plot(new_value={"loss": "weighted"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_boxcar(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'default',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "default"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "1"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '5',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "5"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '3',
                'args': [
                    {'y': [
                        _update_plot(new_value={"boxcar": "3"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


def _get_dataset_model(offset: float = 0) -> dict:
    return {
        'buttons': [
            {
                'method': 'restyle',
                'label': 'default',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "default"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '0',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "0"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "1"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '2',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "2"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '4',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "3"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
            {
                'method': 'restyle',
                'label': '1',
                'args': [
                    {'y': [
                        _update_plot(new_value={"model": "4"})[col] for col in keys
                    ],
                        'name': [key for key in keys]
                    },
                ]
            },
        ],
        'direction': 'down',
        'showactive': True,
        'y': 1 - offset
    }


"""
    Other
"""


def _get_value(new_value: Optional[dict]):
    if new_value is not None:
        assert len(new_value) == 1
        new_key = next(iter(new_value))
        params[new_key] = new_value[new_key]
    for key in user_dict.keys():  # get one by one to see if I get it.
        coincidence = True
        for param_to_match, value_to_match in params.items():
            if user_dict[key]['params'][param_to_match] != value_to_match:
                coincidence = False
                break
        if coincidence:
            return user_dict[key]['data']
    print("WARNING: Match not found, printing default")
    return user_dict[list(user_dict)[0]]['data']


def _update_plot(new_value: Optional[dict]):
    history_path = _get_value(new_value)
    with open(history_path, 'rb') as f:
        saved_history = pickle.load(f)
    df = pd.DataFrame(saved_history)[keys]
    return df


def live_plot():
    df = _update_plot(new_value=None)
    plot_lines = []
    for col in df.columns:
        plot_lines.append(go.Scatter(
            x=df.index, y=df[col], name=col, mode='lines'
        ))
    updatemenus = [
        _get_dataset_mode_dict(),
        _get_dataset_dtype(0.1),
        _get_dataset_library(0.2),
        _get_dataset_dataset_split(0.3),
        _get_dataset_loss(0.4),
        _get_dataset_boxcar(0.5),
        _get_dataset_model(0.6)
    ]

    layout = go.Layout(
        updatemenus=updatemenus
    )
    fig = go.Figure(data=plot_lines, layout=layout)
    fig.show()


if __name__ == '__main__':
    live_plot()

    # img = imread(user_dict['/media/barrachina/data/results/Bretigny/after_icassp/08Friday/run-17h03m24']['image'])
    # fig = px.imshow(img, layout=layout)
    # fig.show()

