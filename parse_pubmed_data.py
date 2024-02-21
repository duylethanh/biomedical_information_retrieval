from tqdm import tqdm
import os
import pubmed_parser as pp
import json
from html import unescape


pubmed_dir = '/vol/tmp/lethanhd/pubmed/baseline'
baseline_dict = {}

id_title_abs = 0
id_title = 0
misc_case = 0
authors_translations = 0
titles_w_brackets = 0

file_list = os.listdir(pubmed_dir)
for filename in tqdm(file_list):
    full_path = os.path.join(pubmed_dir, filename)
    partial_baseline = pp.parse_medline_xml(full_path)

    for article in partial_baseline:
        pmid = article['pmid']

        title = unescape(article['title'])
        # remove brackets in titles like '[Serum immunoglobulin E level in bronchial asthma].'
        # if title.startswith('[') and title.endswith('].'):
        #     if title.endswith("(author's transl)]."):
        #         title = title[1:-20] + '.'
        #         authors_translations += 1
        #     else:
        #         title = title[1:-2] + '.'
        #     titles_w_brackets += 1

        abstract = unescape(article['abstract'])
        # remove multiple white space characters
        abstract = ' '.join(abstract.split())

        # Skip articles without pmid, title or abstract
        if pmid != '' and title != '' and abstract != '':
            baseline_dict[pmid] = [title, abstract]
            id_title_abs += 1
        elif pmid != '' and title != '':
            id_title += 1
        else:
            misc_case += 1


with open('pmid2info.json', 'w') as pmid2info:
    json.dump(baseline_dict, pmid2info, ensure_ascii=False, indent=4)

lines = [
    'paper with pm id, title and abstract ' + str(id_title_abs),
    'paper with pm id and title ' + str(id_title),
    'other cases ' + str(misc_case),
    # 'titles with brackets ', str(titles_w_brackets),
    # 'author translations ', str(authors_translations)
]

with open('pubmed_stats', 'w') as f:
    f.write('\n'.join(lines))

