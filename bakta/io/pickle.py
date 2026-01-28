import pickle
import hashlib
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

def read_pickle(pickle_path: str) -> dict:
    with open(pickle_path, 'rb') as fh:
        pickled_obj = pickle.load(fh)
    return pickled_obj

# TODO: delete if not needed
def feature_to_hash(feature: dict) -> str:
    feature_str = ':'.join([f'{k}_{str(feature[k])}' for k in sorted(feature.keys())])
    feature_hex = hashlib.sha256(feature_str.encode('utf-8')).hexdigest()
    return feature_hex

def feature_nt_to_hash(feature: dict) -> str:
    assert 'nt' in feature
    return hashlib.sha256(feature['nt'].encode('utf-8')).hexdigest()
