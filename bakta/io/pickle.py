import pickle
import hashlib
import logging
import json

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import bakta.constants as bc
import bakta.config as cfg
import bakta.utils as bu


log = logging.getLogger('PICKLE')


def convert_bytes(obj):
    if isinstance(obj, bytes):
        return obj.hex() # aa_hexdigest
    if isinstance(obj, dict):
        return {convert_bytes(k): convert_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_bytes(x) for x in obj]
    return obj

def write_json(data: dict, outpath: Path):
    """
    Write json file with all available data.
    
    Args:
        data: Main data dictionary containing genome info, stats, cds features
        outpath: Output path for pickle file
    """
    
    with outpath.open('w') as fh:
        json.dump(convert_bytes(data), fh, indent=4)

def load_json(inpath: Path, log: logging.Logger) -> dict:
    """
    Load pickle file with all available data.
    
    Args:
        pickle_path: Input path for pickle file
        log: Logger for logging messages
    
    Returns:
        Data dictionary containing genome info, stats, features
    """
    try:
        with open(inpath, 'r') as fh:
            data = json.load(fh)
        return data
    except Exception:
        log.error(f'Failed to load json: {inpath}')
    

def write_pickle_internal(data: dict, pickle_path: Path):
    """
    Write pickle file with all available data.
    
    Args:
        data: Main data dictionary containing genome info, stats, cds features
        pickle_path: Output path for pickle file
    """
    
    with pickle_path.open('wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    

def load_pickle_internal(pickle_path: Path, log: logging.Logger) -> dict:
    """
    Load pickle file with all available data.
    
    Args:
        pickle_path: Input path for pickle file
        log: Logger for logging messages
    
    Returns:
        Data dictionary containing genome info, stats, features
    """
    try:
        with open(pickle_path, 'rb') as fh:
            data = pickle.load(fh)
        return data
    except Exception:
        log.error(f'Failed to load pickle: {pickle_path}')

def write_pickle(data: dict, pickle_path: Path, serializer: str = 'pickle'):
    try:
        cfg.SERIALIZERS[serializer].writer(data, pickle_path)
    except KeyError:
        raise ValueError(f"Unknown serializer: {serializer}")

def load_pickle(pickle_path: Path, log: logging.Logger, serializer: str = 'pickle') -> dict:
    try:
        return cfg.SERIALIZERS[serializer].reader(pickle_path, log)
    except KeyError:
        raise ValueError(f"Unknown serializer: {serializer}")

def _prepare_features(feats: list[dict]) -> list[dict]:
    out = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        f.pop("id", None)
        f.pop("locus", None)
        bu.normalize_coords(f)
        out.append(f)
    return out


def merge_precomputed_cds_and_rna(data: dict, cds_pkl_path, rna_pkl_path, log) -> None:
    """
    Merge precomputed CDS and RNA features from pickle files into main data dictionary.
    
    Args:
        data: Main data dictionary to update
        cds_pkl_path: Path to pickle file containing precomputed CDS features
        rna_pkl_path: Path to pickle file containing precomputed RNA features
        log: Logger for logging messages
    """
    cds_obj = load_pickle(cds_pkl_path, log, cfg.serizalizer)
    rna_obj = load_pickle(rna_pkl_path, log, cfg.serizalizer)

    cds_feats = cds_obj.get('features', [])
    rna_feats = rna_obj.get('features', [])

    rna_feats = [f for f in rna_feats if isinstance(f, dict) and f.get("type") != bc.FEATURE_CDS]
    cds_feats = [f for f in cds_feats if isinstance(f, dict) and f.get("type") == bc.FEATURE_CDS]

    rna_feats = _prepare_features(rna_feats)
    cds_feats = _prepare_features(cds_feats)

    data['features'].extend(cds_feats)
    data['features'].extend(rna_feats)

    log.info(
            "merged features: rna=%d, cds=%d, total=%d",
            len(rna_feats),
            len(cds_feats),
            len(data["features"]),
        )


# TODO: replace everywhere by `load_pickle`
def read_pickle(pickle_path: str) -> dict:
    with open(pickle_path, 'rb') as fh:
        pickled_obj = pickle.load(fh)
    return pickled_obj

def feature_to_hash(feature: dict) -> str:
    feature_str = ':'.join([f'{k}_{str(feature[k])}' for k in sorted(feature.keys())])
    feature_hex = hashlib.sha256(feature_str.encode('utf-8')).hexdigest()
    return feature_hex

# TODO: delete if not needed
def feature_nt_to_hash(feature: dict) -> str:
    assert 'nt' in feature
    return hashlib.sha256(feature['nt'].encode('utf-8')).hexdigest()
