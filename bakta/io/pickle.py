import pickle
import logging

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import bakta.constants as bc
import bakta.config as cfg


log = logging.getLogger('PICKLE')


def write_pickle(data: dict, pickle_path: Path):
    """
    Write comprehensive pickle file with all available data.
    
    Args:
        data: Main data dictionary containing genome info, stats, cds features
        pickle_path: Output path for pickle file
    """
    
    with pickle_path.open('wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    

