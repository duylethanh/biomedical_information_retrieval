import os
import random
import hashlib
import json
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer


random.seed(42)

pmc_dirs = [
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC000xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC001xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC002xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC003xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC004xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC005xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC006xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC007xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC008xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC009xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC010xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC001xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC002xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC003xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC004xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC005xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC006xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC007xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC008xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC009xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_noncomm/xml/PMC010xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC000xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC001xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC002xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC003xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC004xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC005xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC006xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC007xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC008xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC009xxxxxx',
    '/vol/tmp/lethanhd/pmc/oa_other/xml/PMC010xxxxxx',
]

pubmed_dir = '/glusterfs/dfs-gfs-dist/lethanhd/pubmed/baseline'
qid2info_path = '/vol/tmp/lethanhd/data_parser/50k_iterative_improved/qid2info.json'
pmid2info_path = '/vol/tmp/lethanhd/data_parser/pmid2info.json'
stats_dir = '/vol/tmp/lethanhd/data_parser/analysis_stats/50k_iterative_improved'


def verify_pubmed_files(pubmed_dir):
    correct = 0
    corrupted = 0
    files = os.listdir(pubmed_dir)
    for file in tqdm(files):
        file_path = os.path.join(pubmed_dir, file)
        if file_path.endswith('.gz'):
            with open(file_path, "rb") as f:
                readable_hash = hashlib.md5(bytes).hexdigest()

            with open(file_path + '.md5', 'r') as f:
                md5sum = f.read().split()[-1]

            if readable_hash == md5sum:
                correct += 1
            else:
                corrupted += 1

    print('num correct files:', correct)
    print('num corrupted files:', corrupted)


def random_sample_pmc_files(num_samples=30000):
    files = []
    for pmc_dir in tqdm(pmc_dirs):
        files.extend([os.path.join(pmc_dir, file) for file in os.listdir(pmc_dir)])

    random_samples = random.sample(range(len(files)), k=num_samples)

    with open(os.path.join(stats_dir, 'random_file_samples_10k'), 'w') as f:
        f.write('\n'.join([files[sample] for sample in random_samples]))


def analyze_qid2info(qid2info_path, tokenizer, stats_dir):

    with open(qid2info_path, 'r') as f:
        qid2info = json.load(f)

    query_lengths = {}
    leq_64 = 0
    leq_128 = 0
    leq_192 = 0
    leq_256 = 0
    for entry in tqdm(qid2info, position=0, leave=True):
        query = qid2info[entry]
        len_query_toks = len(tokenizer.encode(query))

        if len_query_toks in query_lengths:
            query_lengths[len_query_toks] += 1
        else:
            query_lengths[len_query_toks] = 1

        if len_query_toks < 65:
            leq_64 += 1
        if len_query_toks < 129:
            leq_128 += 1
        if len_query_toks < 193:
            leq_192 += 1
        if len_query_toks < 257:
            leq_256 += 1

    with open(os.path.join(stats_dir, 'query_lengths'), 'w') as f:
        json.dump(query_lengths, f, indent=4)

    return [
        'path: ' + str(qid2info_path),
        'num queries: ' + str(len(qid2info)),
        'query tokenizations less equal than 64: ' + '{} ({})'.format(str(round(leq_64/len(qid2info), 4)), leq_64),
        'query tokenizations less equal than 128: ' + '{} ({})'.format(str(round(leq_128/len(qid2info), 4)), leq_128),
        'query tokenizations less equal than 192: ' + '{} ({})'.format(str(round(leq_192/len(qid2info), 4)), leq_192),
        'query tokenizations less equal than 256: ' + '{} ({})'.format(str(round(leq_256/len(qid2info), 4)), leq_256),
    ]


def analyze_pmid2info(pmid2info_path, tokenizer, stats_dir):

    with open(pmid2info_path, 'r') as f:
        pmid2info = json.load(f)

    abstract_lengths = {}
    leq_512 = 0
    leq_768 = 0
    leq_1024 = 0
    leq_1280 = 0
    for entry in tqdm(pmid2info, position=0, leave=True):
        abstract = pmid2info[entry][1]
        title = pmid2info[entry][0]
        len_doc_tokens = len(tokenizer.encode(abstract)) + len(tokenizer.encode(title))
        
        if len_doc_tokens in abstract_lengths:
            abstract_lengths[len_doc_tokens] += 1
        else:
            abstract_lengths[len_doc_tokens] = 1
        
        if len_doc_tokens < 513:
            leq_512 += 1
        if len_doc_tokens < 769:
            leq_768 += 1
        if len_doc_tokens < 1025:
            leq_1024 += 1
        if len_doc_tokens < 1281:
            leq_1280 += 1

    with open(os.path.join(stats_dir, 'doc_infos'), 'w') as f:
        json.dump(abstract_lengths, f, indent=4)

    return [
        'path: ' + str(pmid2info_path),
        'num abstracts: ' + str(len(pmid2info)),
        'abstract+title tokenizations less equal than 512: ' + '{} ({})'.format(str(round(leq_512 / len(pmid2info), 4)), leq_512),
        'abstract+title tokenizations less equal than 768: ' + '{} ({})'.format(str(round(leq_768 / len(pmid2info), 4)), leq_768),
        'abstract+title tokenizations less equal than 1024: ' + '{} ({})'.format(str(round(leq_1024 / len(pmid2info), 4)), leq_1024),
        'abstract+title tokenizations less equal than 1280: ' + '{} ({})'.format(str(round(leq_1280 / len(pmid2info), 4)), leq_1280),
    ]


def analyze_data(qid2info_path, pmid2info_path, stats_dir, do_analyze_qid2info=False, do_analyze_pmid2info=False):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    output_lines = []

    if do_analyze_qid2info:
        output_lines.extend(analyze_qid2info(qid2info_path, tokenizer, stats_dir))

    if do_analyze_pmid2info:
        output_lines.extend(analyze_pmid2info(pmid2info_path, tokenizer, stats_dir))

    with open(os.path.join(stats_dir, 'tokenization_lengths'), 'w') as f:
        f.write('\n'.join(output_lines))


# verify_pubmed_files(pubmed_dir)
# random_sample_pmc_files(10000)
# analyze_data(qid2info_path=qid2info_path, pmid2info_path=pmid2info_path)

# analyze_data(qid2info_path='/vol/tmp/lethanhd/data_parser/50k_iterative_improved/qid2info.json', pmid2info_path=pmid2info_path, stats_dir = '/vol/tmp/lethanhd/data_parser/analysis_stats/50k_iterative_improved')

