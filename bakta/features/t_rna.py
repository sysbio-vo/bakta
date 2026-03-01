import logging
import subprocess as sp

from collections import OrderedDict
from pathlib import Path

from Bio import SeqIO

import bakta.config as cfg
import bakta.constants as bc
import bakta.so as so
import bakta.utils as bu


log = logging.getLogger('T_RNA')


AMINO_ACID_DICT = {
    'ala': ('A', so.SO_TRNA_ALA),
    'gln': ('Q', so.SO_TRNA_GLN),
    'glu': ('E', so.SO_TRNA_GLU),
    'gly': ('G', so.SO_TRNA_GLY),
    'pro': ('P', so.SO_TRNA_PRO),
    'met': ('M', so.SO_TRNA_MET),
    'fmet':('fM', so.SO_TRNA_MET),
    'asp': ('D', so.SO_TRNA_ASP),
    'thr': ('T', so.SO_TRNA_THR),
    'val': ('V', so.SO_TRNA_VAL),
    'tyr': ('Y', so.SO_TRNA_TYR),
    'cys': ('C', so.SO_TRNA_CYS),
    'ile': ('I', so.SO_TRNA_ILE),
    'ile2':('I', so.SO_TRNA_ILE),
    'ser': ('S', so.SO_TRNA_SER),
    'leu': ('L', so.SO_TRNA_LEU),
    'trp': ('W', so.SO_TRNA_TRP),
    'lys': ('K', so.SO_TRNA_LYS),
    'asn': ('N', so.SO_TRNA_ASN),
    'arg': ('R', so.SO_TRNA_ARG),
    'his': ('H', so.SO_TRNA_HIS),
    'phe': ('F', so.SO_TRNA_PHE),
    'sec': ('U', so.SO_TRNA_SELCYS)
}


def predict_t_rnas(data: dict, sequences_path: Path):
    """Search for tRNA sequences."""

    txt_output_path = cfg.tmp_path.joinpath('trna.tsv')
    fasta_output_path = cfg.tmp_path.joinpath('trna.fasta')
    cmd = [
        'tRNAscan-SE',
        '-B',
        '--output', str(txt_output_path),
        '--fasta', str(fasta_output_path),
        '--thread', str(cfg.threads),
        str(sequences_path)
    ]
    log.debug('cmd=%s', cmd)
    log.info(f'running {cmd} for trnas prediction')
    proc = sp.run(
        cmd,
        cwd=str(cfg.tmp_path)
    )

    log.info(f'finished {cmd} for trnas prediction')
    if(proc.returncode != 0):
        log.debug('stdout=\'%s\', stderr=\'%s\'', proc.stdout, proc.stderr)
        log.warning('tRNAs failed! tRNAscan-SE-error-code=%d', proc.returncode)
        raise Exception(f'tRNAscan-SE error! error code: {proc.returncode}')

    log.info(f'collecting trnas')
    trnas = {}
    sequences = {seq['id']: seq for seq in data['sequences']}

    log.info(f'opening {txt_output_path}')
    with txt_output_path.open() as fh:
        for line in fh.readlines()[3:]:  # skip first 3 lines
            (sequence_id, trna_id, start, stop, trna_type, anti_codon, intron_begin, bounds_end, score, note) = line.split('\t')
            log.info(f'processing {(sequence_id, trna_id, start, stop, trna_type, anti_codon, intron_begin, bounds_end, score, note)}')

            start, stop, strand = int(start), int(stop), bc.STRAND_FORWARD
            if(start > stop):  # reverse
                start, stop = stop, start
                strand = bc.STRAND_REVERSE
            sequence_id = sequence_id.strip()  # bugfix for extra single whitespace in tRNAscan-SE output

            trna = OrderedDict()
            trna['type'] = bc.FEATURE_T_RNA
            trna['sequence'] = sequence_id
            trna['start'] = start
            trna['stop'] = stop
            trna['strand'] = strand
            trna['gene'] = None
            trna['product'] = 'tRNA-Xxx'
            if(trna_type != 'Undet' and trna_type != 'Sup'):
                log.info(f'getting AA code')
                aa_code = AMINO_ACID_DICT.get(trna_type.lower(), ('', None))[0]
                log.info(f'{aa_code = }')
                trna['gene'] = f'trn{aa_code}'
                trna['product'] = f'tRNA-{trna_type}({anti_codon.lower()})'
                trna['amino_acid'] = trna_type
                trna['anti_codon'] = anti_codon.lower()

            if('pseudo' in note):
                log.info(f'trna feature is a pseudogene')
                trna[bc.PSEUDOGENE] = True

            trna['score'] = float(score)
            log.info(f'currrent trna: {trna = }')
            log.info(f'extracting feature sequence')
            nt = bu.extract_feature_sequence(trna, sequences[sequence_id])  # extract nt sequences
            trna['nt'] = nt

            log.info(f'adding dbxrefs')
            trna['db_xrefs'] = []
            so_term = AMINO_ACID_DICT.get(trna_type.lower(), ('', None))[1]
            if(so_term):
                trna['db_xrefs'].append(so_term.id)

            log.info(f'adding entry to trna dict')
            key = f'{sequence_id}.trna{trna_id}'
            log.info(f'{key = }')
            trnas[key] = trna
            log.info(f'added entry to tnra dict')
            log.info(
                'seq=%s, start=%i, stop=%i, strand=%s, gene=%s, product=%s, score=%1.1f, nt=[%s..%s]',
                trna['sequence'], trna['start'], trna['stop'], trna['strand'], trna.get('gene', ''), trna['product'], trna['score'], nt[:10], nt[-10:]
            )

    with fasta_output_path.open() as fh:
        for record in SeqIO.parse(fh, 'fasta'):
            trna = trnas[record.id]
            nt = str(record.seq).upper()
            if('anti_codon' in trna and trna['amino_acid'].lower() not in ['fmet', 'ile2', 'sec', 'sup']):  # exclude fMet, Ile2 and Sec (INSDC wrong anticodon issue)
                anticodon_pos = trna['nt'].lower().find(trna['anti_codon'])
                if(anticodon_pos > -1):
                    if(trna['strand'] == bc.STRAND_FORWARD):
                        start = trna['start'] + anticodon_pos
                        stop = start + 2
                    else:
                        stop = trna['stop'] - anticodon_pos
                        start = stop - 2
                    trna['anti_codon_pos'] = (start, stop)
    trnas = list(trnas.values())
    log.info('predicted=%i', len(trnas))
    return trnas
