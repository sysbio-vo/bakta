import atexit
import logging
import os
import sys

from datetime import datetime
from typing import Sequence
from pathlib import Path

import bakta
import bakta.constants as bc
import bakta.config as cfg
import bakta.db as db
import bakta.utils as bu
import bakta.io.pickle as pickle
import bakta.expert.amrfinder as exp_amr
import bakta.expert.protein_sequences as exp_aa_seq
import bakta.expert.protein_hmms as exp_aa_hmms
import bakta.features.annotation as anno
import bakta.features.orf as orf
import bakta.features.cds as feat_cds
import bakta.io.fasta as fasta
import bakta.io.json as json
import bakta.io.tsv as tsv
import bakta.ups as ups
import bakta.ips as ips
import bakta.psc as psc
import bakta.pscc as pscc

log = logging.getLogger('PSEUDOGENES')

def main_bulk():
    # parse options and arguments
    parser = bu.init_parser(sub_command='_pseudo_bulk')
    parser.add_argument('manifest', metavar='<input>', help='Manifest file with paths to pickled Bakta data objects with annotated CDSs to detect pseudogenes')
    
    arg_group_io = parser.add_argument_group('Input / Output')
    arg_group_io.add_argument('--db', '-d', action='store', default=None, help='Database path (default = <bakta_path>/db). Can also be provided as BAKTA_DB environment variable.')
    arg_group_io.add_argument('--output', '-o', action='store', default=os.getcwd(), help='Output directory (default = current working directory)')
    arg_group_io.add_argument('--prefix', '-p', action='store', default=None, help='Prefix for output files')
    arg_group_io.add_argument('--force', '-f', action='store_true', help='Force overwriting existing output folder')
        
    arg_group_general = parser.add_argument_group('General')
    arg_group_general.add_argument('--help', '-h', action='help', help='Show this help message and exit')
    arg_group_general.add_argument('--verbose', '-v', action='store_true', help='Print verbose information')
    arg_group_general.add_argument('--debug', action='store_true', help='Run Bakta in debug mode. Temp data will not be removed.')
    arg_group_general.add_argument('--threads', '-t', action='store', type=int, default=0, help='Number of threads to use (default = number of available CPUs)')
    arg_group_general.add_argument('--tmp-dir', action='store', default=None, dest='tmp_dir', help='Location for temporary files (default = system dependent auto detection)')
    arg_group_general.add_argument('--version', '-V', action='version', version=f'%(prog)s {cfg.version}')
    args = parser.parse_args()

    ############################################################################
    # Setup logging
    ############################################################################
    cfg.prefix = args.prefix if args.prefix else Path(args.input).stem
    output_path = cfg.check_output_path(args.output, args.force)
    cfg.force = args.force
    log.info('force=%s', args.force)
    
    bu.setup_logger(output_path, cfg.prefix, args)
    log.info('prefix=%s', cfg.prefix)
    log.info('output=%s', output_path)

    try:
        if(args.manifest == ''):
            raise ValueError('Path to a manifest must be non-empty')
        manifest_path = Path(args.manifest).resolve()
    except:
        log.error('provided input CDS data file not valid! path=%s', args.manifest)
        sys.exit(f'ERROR: input CDS data file ({args.manifest}) not valid!')

    log.info('manifest-path=%s', manifest_path)

    cfg.check_db_path(args)
    cfg.db_info = db.check(cfg.db_path)
    cfg.check_tmp_path(args)
    cfg.check_threads(args)
    cfg.debug = args.debug
    log.info('debug=%s', cfg.debug)
    cfg.verbose = True if cfg.debug else args.verbose
    log.info('verbose=%s', cfg.verbose)

    cfg.translation_table = 11 # TODO: set from config

    # bu.test_dependencies() # TODO: determine required depdendencies for pseudogene prediction
    if(cfg.verbose):
        print(f'Bakta v{cfg.version}')
        print('Options and arguments:')
        print(f'\tinput: {manifest_path}')
        print(f"\tdb: {cfg.db_path}, version {cfg.db_info['major']}.{cfg.db_info['minor']}")
        print(f'\toutput: {cfg.output_path}')
        if(cfg.force): print(f'\tforce: {cfg.force}')
        print(f'\ttmp directory: {cfg.tmp_path}')
        print(f'\tprefix: {cfg.prefix}')
        print(f'\t# threads: {cfg.threads}')

    if(cfg.debug):
        print(f"\nBakta runs in DEBUG mode! Temporary data will not be destroyed at: {cfg.tmp_path}")
    else:
        atexit.register(bu.cleanup, log, cfg.tmp_path)  # register cleanup exit hook

    print('\nStart bulk pseudogene prediction...')
    print('\nRead manifest file...')

    sample_to_data = {}
    with open(manifest_path, 'r') as handle:
        bakta_pickles = handle.readlines()
        for bakta_pickle_path in bakta_pickles:
            bakta_pickle_path = bakta_pickle_path.rstrip('\n')
            print(f'\nExtracting pseudogene candidates from sample {bakta_pickle_path}...')
            data = pickle.read_pickle(bakta_pickle_path)
            sample_to_data[bakta_pickle_path] = data

    # get all hypotheticals and save intermediate files to tmp dir
    print('\Find pseudogene candidates...')
    bulk_candidates, seq_to_feature, sample_to_feature, seq_to_sample, candidate_to_sample_id = get_bulk_candidates(sample_to_data)

    candidates_search_end_time = datetime.now()
    candidates_detection_duration = (candidates_search_end_time - cfg.run_start).total_seconds()
    print(f'Pseudogene candidates search finished in {int(candidates_detection_duration / 60):01}:{int(candidates_detection_duration % 60):02} [mm:ss].')

    bulk_pseudogenes = predict_bulk_pseudogenes(bulk_candidates, seq_to_sample, sample_to_data, seq_to_feature, candidate_to_sample_id) if len(bulk_candidates) > 0 else []

    cfg.run_end = datetime.now()
    pseudogene_predicition_duration = (cfg.run_end - candidates_search_end_time).total_seconds()
    print(f'Pseudogene prediction finished in {int(pseudogene_predicition_duration / 60):01}:{int(pseudogene_predicition_duration % 60):02} [mm:ss].')

    # write updated Bakta dictionaries
    print(f'Saving updated Bakta CDS features')
    for input_filepath, updated_data in sample_to_data.items():
        filepath_basename = os.path.splitext(os.path.basename(input_filepath))[0]
        output_pickle_path = cfg.output_path.joinpath(f'{cfg.prefix}.{filepath_basename}.with_pseudogenes.pkl')
        print(f'\nExport pseudogene search results to: {output_pickle_path}')
        pickle.write_pickle(updated_data, output_pickle_path)

    total_search_time = (cfg.run_end - cfg.run_start).total_seconds()
    print(f'Total elapsed time: {int(total_search_time / 60):01}:{int(total_search_time % 60):02} [mm:ss].')

        

from collections import defaultdict

def get_bulk_candidates(sample_to_data: dict[str, object]):
    """
    Extract all pseudogene candidates from a set of samples and combined them into a single Bakta data object
    """

    unique_aa_seqs = set()
    unique_hypothetical_features = []
    seq_to_feature = defaultdict(list) # aa hexdist used as key
    sample_to_feature = defaultdict(list) # sample path used as key

    # this is sort of a many-to-many relation
    seq_to_sample = defaultdict(list) # sequence hexdigest will be used as key and sample identifier (absolute path) as value
    feature_to_sample_id = defaultdict(list) # NOTE: in theory is it possible to have identical features for different samples?

    # get all hypotheticals
    for bakta_pickle_path, data in sample_to_data.items():

        for feat in data['features']:
            if feat['type'] != bc.FEATURE_CDS or 'hypothetical' not in feat or 'edge' in feat or feat.get('start_type', 'Edge') == 'Edge':
                continue
            sample_to_feature[bakta_pickle_path].append(feat) # will be used to elongate sequences during pseudogene prediction
            seq_to_feature[feat['aa_hexdigest']].append(feat) # will be modified in place during pseudogene prediction
            seq_to_sample[feat['aa_hexdigest']].append(bakta_pickle_path)
            feature_to_sample_id[pickle.feature_to_hash(feat)].append(bakta_pickle_path)

            # unique set of proteins that will be used for pseudogene candidate prediction
            if feat['aa_hexdigest'] not in unique_aa_seqs:
                unique_hypothetical_features.append(feat)
                unique_aa_seqs.add(feat['aa_hexdigest'])

    # return unique_hypothetical_features, seq_to_feature, sample_to_feature
    
    bulk_candidates, candidate_to_sample_id = feat_cds.predict_pseudo_candidates_bulk(unique_hypothetical_features, feature_to_sample_id)
    # bulk_candidates = []

    pickle.write_pickle(bulk_candidates, cfg.tmp_path.joinpath('cds.pseudo.candidates_bulk_list.pkl')) # DEBUG
    pickle.write_pickle(candidate_to_sample_id, cfg.tmp_path.joinpath('cds.pseudo.candidate_to_sample_id.pkl')) # DEBUG
    pickle.write_pickle(seq_to_feature, cfg.tmp_path.joinpath('seq_to_feature.pkl')) # DEBUG
    pickle.write_pickle(sample_to_feature, cfg.tmp_path.joinpath('sample_to_feature.pkl')) # DEBUG

    return bulk_candidates, seq_to_feature, sample_to_feature, seq_to_sample, candidate_to_sample_id
            
def predict_bulk_pseudogenes(bulk_candidates, seq_to_sample, sample_to_data, seq_to_feature, candidate_to_sample_id):
    pseudogenes_bulk = feat_cds.detect_pseudogenes_bulk(bulk_candidates, seq_to_sample, sample_to_data, seq_to_feature, candidate_to_sample_id)
    psc.lookup(pseudogenes_bulk, pseudo=True)
    pscc.lookup(pseudogenes_bulk, pseudo=True)
    for pseudogene in pseudogenes_bulk:
        anno.combine_annotation(pseudogene)
    print(f'\t\tverified bulk pseudogenes: {len(pseudogenes_bulk)}')
    return pseudogenes_bulk

def main():
    # parse options and arguments
    parser = bu.init_parser(sub_command='_pseudo')
    parser.add_argument('input', metavar='<input>', help='Path to a pickled Bakta data object with annotated CDS to detect pseudogenes')
    
    arg_group_io = parser.add_argument_group('Input / Output')
    arg_group_io.add_argument('--db', '-d', action='store', default=None, help='Database path (default = <bakta_path>/db). Can also be provided as BAKTA_DB environment variable.')
    arg_group_io.add_argument('--output', '-o', action='store', default=os.getcwd(), help='Output directory (default = current working directory)')
    arg_group_io.add_argument('--prefix', '-p', action='store', default=None, help='Prefix for output files')
    arg_group_io.add_argument('--force', '-f', action='store_true', help='Force overwriting existing output folder')
        
    arg_group_general = parser.add_argument_group('General')
    arg_group_general.add_argument('--help', '-h', action='help', help='Show this help message and exit')
    arg_group_general.add_argument('--verbose', '-v', action='store_true', help='Print verbose information')
    arg_group_general.add_argument('--debug', action='store_true', help='Run Bakta in debug mode. Temp data will not be removed.')
    arg_group_general.add_argument('--threads', '-t', action='store', type=int, default=0, help='Number of threads to use (default = number of available CPUs)')
    arg_group_general.add_argument('--tmp-dir', action='store', default=None, dest='tmp_dir', help='Location for temporary files (default = system dependent auto detection)')
    arg_group_general.add_argument('--version', '-V', action='version', version=f'%(prog)s {cfg.version}')
    args = parser.parse_args()

    ############################################################################
    # Setup logging
    ############################################################################
    cfg.prefix = args.prefix if args.prefix else Path(args.input).stem
    output_path = cfg.check_output_path(args.output, args.force)
    cfg.force = args.force
    log.info('force=%s', args.force)
    
    bu.setup_logger(output_path, cfg.prefix, args)
    log.info('prefix=%s', cfg.prefix)
    log.info('output=%s', output_path)


    try:
        if(args.input == ''):
            raise ValueError('Pickled CDS data file path argument must be non-empty')
        cds_features_path = Path(args.input).resolve()
        print(f'{cds_features_path = }')
        cfg.check_readability('annotated CDS bakta dictionary', cds_features_path)
        cfg.check_content_size('annotated CDS bakta dictionary', cds_features_path)
    except:
        log.error('provided input CDS data file not valid! path=%s', args.input)
        sys.exit(f'ERROR: input CDS data file ({args.input}) not valid!')
    log.info('input-path=%s', cds_features_path)


    cfg.check_db_path(args)
    cfg.db_info = db.check(cfg.db_path)
    cfg.check_tmp_path(args)
    cfg.check_threads(args)
    cfg.debug = args.debug
    log.info('debug=%s', cfg.debug)
    cfg.verbose = True if cfg.debug else args.verbose
    log.info('verbose=%s', cfg.verbose)

    cfg.translation_table = 11 # TODO: set from config

    # bu.test_dependencies() # TODO: determine required depdendencies for pseudogene prediction
    if(cfg.verbose):
        print(f'Bakta v{cfg.version}')
        print('Options and arguments:')
        print(f'\tinput: {cds_features_path}')
        print(f"\tdb: {cfg.db_path}, version {cfg.db_info['major']}.{cfg.db_info['minor']}")
        print(f'\toutput: {cfg.output_path}')
        if(cfg.force): print(f'\tforce: {cfg.force}')
        print(f'\ttmp directory: {cfg.tmp_path}')
        print(f'\tprefix: {cfg.prefix}')
        print(f'\t# threads: {cfg.threads}')


    if(cfg.debug):
        print(f"\nBakta runs in DEBUG mode! Temporary data will not be destroyed at: {cfg.tmp_path}")
    else:
        atexit.register(bu.cleanup, log, cfg.tmp_path)  # register cleanup exit hook

    ############################################################################
    # Import annotated CDS from pickled objects
    ############################################################################
    try:
        print('Read pickled dictionary with annotated CDS...')
        data = pickle.read_pickle(cds_features_path)
        log.info('imported CDS sequences=%i', len(data['features']))
        print(f"\timported: {len(data['features'])}")
    except:
        log.error('wrong file format or other issues with the input file!', exc_info=True)
        sys.exit('ERROR: wrong file format or other issues with the input file!')


    print('\nStart pseudogene prediction...')
    cdss = [feat for feat in data['features'] if feat['type'] == bc.FEATURE_CDS]
    _ = predict_pseudogenes_from_cdss(data, cdss, log)

    cfg.run_end = datetime.now()
    run_duration = (cfg.run_end - cfg.run_start).total_seconds()

    ############################################################################
    # Write output files
    # - write updated Bakta data object with predicted pseudogenes
    # - remove temp directory
    ############################################################################
    output_pickle_path = cfg.output_path.joinpath(f'{cfg.prefix}.with_pseudogenes.pkl')
    print(f'\nExport pseudogene detection results to: {output_pickle_path}')
    pickle.write_pickle(data, output_pickle_path)

    print(f'Pseudogene annotation successfully finished in {int(run_duration / 60):01}:{int(run_duration % 60):02} [mm:ss].')


def predict_pseudogenes_from_cdss(data: dict, cdss: list, log: logging.Logger) -> list:
    hypotheticals = [cds for cds in cdss if 'hypothetical' in cds and 'edge' not in cds and cds.get('start_type', 'Edge') != 'Edge']
    print(f'Found {len(hypotheticals)} hypothetical CDS')
    if(len(hypotheticals) > 0  and  not cfg.skip_pseudo):
        if(cfg.db_info['type'] == 'full'):
            print('\tdetect pseudogenes...')
            log.debug('search pseudogene candidates')
            pseudo_candidates = feat_cds.predict_pseudo_candidates(hypotheticals)
            print(f'\t\tcandidates: {len(pseudo_candidates)}')
            pseudogenes = feat_cds.detect_pseudogenes(pseudo_candidates, cdss, data) if len(pseudo_candidates) > 0 else []
            psc.lookup(pseudogenes, pseudo=True)
            pscc.lookup(pseudogenes, pseudo=True)
            for pseudogene in pseudogenes:
                anno.combine_annotation(pseudogene)
            print(f'\t\tverified: {len(pseudogenes)}')
            return pseudogenes
        else:
            print(f'\tskip pseudogene detection with light db version')
    return []


if __name__ == '__main__':
    main()