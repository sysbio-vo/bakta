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