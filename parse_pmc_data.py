from lxml import etree
from tqdm import tqdm
from html import unescape
import spacy
import json
import os
import re
import time
import argparse

spacy.require_gpu()

print('Start loading sentence splitter')
sentence_segmenter = spacy.load('en_core_sci_scibert')
print('Finished loading sentence splitter')


def get_reference_content_id(reference_content):
    global num_doi_refs, num_pubmed_refs

    if reference_content.find('pub-id[@pub-id-type="pmid"]') is not None:
        num_pubmed_refs += 1
        return reference_content.find('pub-id[@pub-id-type="pmid"]').text
    # ToDo look up cited documents by doi ID
    # elif reference_content.find('pub-id[@pub-id-type="doi"]') is not None:
    #    num_doi_refs += 1
    #    return reference_content.find('pub-id[@pub-id-type="doi"]').text

    return None


def get_reference_dict(file_path):
    global non_usable_reflist, papers_wo_reflist, num_doi_refs, num_pubmed_refs, cit_wo_pubid_doiid, successful_reflist, unsuccesful_papers

    try:
        paper = etree.parse(file_path)
    except Exception as e:
        unsuccesful_papers.append((file_path, 'etree parse failed in get_reference_dict', repr(e)))
        return

    try:
        pmc_id = paper.xpath('.//front//article-meta//article-id[@pub-id-type="pmc"]')[0].text
    except IndexError:
        pmc_id = file_path.split('/')[-1].split('.')[0]

    ref_list = paper.xpath('.//ref-list//ref[@id]')
    if len(ref_list) == 0:
        papers_wo_reflist.append(pmc_id)
        return

    paper_ref_dict = {}

    for ref in ref_list:
        # for ref in tqdm(ref_list):
        if ref.tag == 'ref':
            ref_id = ref.attrib['id']
            # print(ref_id)

            if ref.find('citation-alternatives') is not None:
                ref = ref.find('citation-alternatives')

            if ref.find('element-citation') is not None:
                pubmed_id = get_reference_content_id(ref.find('element-citation'))

                if pubmed_id is not None:
                    paper_ref_dict[ref_id] = pubmed_id
                    continue

            if ref.find('citation') is not None:
                pubmed_id = get_reference_content_id(ref.find('citation'))

                if pubmed_id is not None:
                    paper_ref_dict[ref_id] = pubmed_id
                    continue

            if ref.find('mixed-citation') is not None:
                pubmed_id = get_reference_content_id(ref.find('mixed-citation'))
                if pubmed_id is not None:
                    paper_ref_dict[ref_id] = pubmed_id
                    continue

            cit_wo_pubid_doiid.append((pmc_id, ref_id))
            continue

    if len(paper_ref_dict.keys()) == 0:
        non_usable_reflist.append(pmc_id)
    else:
        successful_reflist += 1
        return paper_ref_dict


def cleanse_sentence(sent):
    sent = unescape(sent)  # unescape html

    sent = re.sub(r'<(?![/]?xref)[^<]*?>', '', sent)  # remove non-xref tags
    sent = re.sub(r'</xref>[^-;,]?[-;,][^-;,]?<xref', '</xref>,<xref', sent)  # replace citation ranges by comma separation
    sent = re.sub(r'</xref>\w{0,2}\)', '</xref>)', sent)  # remove sub-citation e.g. ""<ref>A"

    sent = sent.replace('(e.g., <xref', '(<xref')
    sent = sent.replace('(see <xref', '(<xref')
    sent = sent.replace('</xref> and <xref', '</xref>,<xref')
    sent = sent.replace('</xref>, and <xref', '</xref>,<xref')

    return sent


def split_sentence_at_citations(sent):
    # split citations by delimiter [<xref, (<xref or white-space
    sent_array = re.split(r'\[(<xref .+?>.+?</xref>)\]', sent)

    first_partial_split_text = []
    for part in sent_array:
        first_partial_split_text.extend(re.split(r'\((<xref .+?>.+?</xref>)\)', part))

    second_partial_split_text = []
    for part in first_partial_split_text:
        second_partial_split_text.extend(re.split(r' (<xref .+?>.+?</xref>)', part))

    # handle cases, where spacy split sentences mid citation enumeration, this captures the first part
    third_partial_split_text = []
    for part in second_partial_split_text:
        third_partial_split_text.extend(re.split(r'[([](<xref[^>]+>[^<]+</xref>(?:,<xref[^>]+>[^<]+</xref>)*)[;,]', part))

    final_split_text = []
    for part in third_partial_split_text:
        # make sure to only split unprocessed sentences, i.e. they dont start with <xref
        if not part.startswith('<xref') and '<xref' in part:
            part = re.sub(r'</xref>[-;,]?\s?<xref', '</xref><xref', part)
            final_split_text.extend(re.split(r'(<xref[^>]+>[^<]+</xref>)', part))
        else:
            final_split_text.append(part)

    return final_split_text


def handle_split_text(split_text, pmc_id, sent, ref_dict, pmid2info, method='iterative'):
    global lines_for_json

    query = ''
    xref_ids = []
    query_article_pairs_in_sent = []
    for part in split_text:
        if '</xref>' not in part:
            query += part.rstrip()
        else:
            # filter out citations for empty queries
            if query == '':
                continue

            # handle citation enumeration
            if '</xref>,<xref' in part:
                cit_list = part.split(',')
                for citation in cit_list:
                    try:
                        xref = etree.fromstring(citation)
                    except Exception as e:
                        # print(part, repr(e))
                        lines_for_json.append(pmc_id)
                        lines_for_json.append('\nsentence: ' + sent)
                        lines_for_json.append('\npart: ' + citation)
                        lines_for_json.append('\nerror: ' + str(repr(e)))
                        lines_for_json.append('\n\n')

                        if method == 'iterative':
                            # return already generated pairs
                            return query_article_pairs_in_sent
                        else:
                            # skip whole sentence
                            return

                            # skip non-bibr "citations"
                    ref_type = xref.attrib['ref-type']
                    if ref_type == 'fig' or ref_type == 'table' or ref_type == 'supplementary-material':
                        continue

                    xref_ids.append(xref.attrib['rid'])
            else:
                try:
                    xref = etree.fromstring(part)
                except Exception as e:
                    lines_for_json.append(pmc_id)
                    lines_for_json.append('\nsentence: ' + sent)
                    lines_for_json.append('\npart: ' + part)
                    lines_for_json.append('\nerror: ' + str(repr(e)))
                    lines_for_json.append('\n\n')

                    if method == 'iterative':
                        # return already generated pairs
                        return query_article_pairs_in_sent
                    else:
                        # skip whole sentence
                        return

                # skip non-bibr "citations"
                ref_type = xref.attrib['ref-type']
                if ref_type == 'fig' or ref_type == 'table' or ref_type == 'supplementary-material':
                    continue

                xref_ids.append(xref.attrib['rid'])

            if method == 'iterative':
                query_article_pairs_in_sent.append(generate_query_article_pairs_from_lists(pmid2info, ref_dict, query, xref_ids))

                xref_ids = []

    if method == 'total':
        query_article_pairs_in_sent.append(generate_query_article_pairs_from_lists(pmid2info, ref_dict, query, xref_ids))

    return query_article_pairs_in_sent


def generate_query_article_pairs_from_lists(pmid2info, ref_dict, query, xref_ids):
    return query, [ref_dict[xref_id] for xref_id in xref_ids if xref_id in ref_dict and str(ref_dict[xref_id]) in pmid2info]


def get_query_article_pairs(file_path, ref_dict, pmid2info, method='iterative'):
    global unsuccesful_papers, lines_for_json

    try:
        paper = etree.parse(file_path)
    except Exception as e:
        unsuccesful_papers.append((file_path, 'etree parse failed in get_query_article_pairs', repr(e)))
        return

    try:
        pmc_id = paper.xpath('.//front//article-meta//article-id[@pub-id-type="pmc"]')[0].text
    except IndexError:
        pmc_id = file_path.split('/')[-1].split('.')[0]

    paragraphs = paper.xpath('//body//p')

    query_article_pairs = []
    for paragraph in paragraphs:
        paragraph_text = etree.tostring(paragraph).decode('us-ascii')

        # remove opening and closing p-tag
        paragraph_text = paragraph_text.replace('</p>', '')
        paragraph_text = re.sub(r'<p xmlns.+?>', '', paragraph_text)  # remove xml declaration

        # split paragraph into sentences using spacy
        try:
            paragraph_spacy_sentences = [str(sentence) for sentence in sentence_segmenter(paragraph_text).sents]
        except Exception as e:
            lines_for_json.append(pmc_id)
            lines_for_json.append('\nparagraph: ' + paragraph_text)
            lines_for_json.append('\nparagrapherror: ' + str(repr(e)))
            lines_for_json.append('\n\n')
            continue

        for sent in paragraph_spacy_sentences:

            # skip sentences without bibr citations
            if 'ref-type="bibr"' not in sent:
                continue

            sent = cleanse_sentence(sent)

            # filter out sentences that only consist of a citation and no query
            if re.match(r'^[\[(]?<xref .*?>.+?</xref>[\])]?$', sent):
                # print(sent)
                continue

            split_text = split_sentence_at_citations(sent)

            qa_pairs_in_sent = handle_split_text(split_text, pmc_id, sent, ref_dict=ref_dict, pmid2info=pmid2info, method=method)

            if qa_pairs_in_sent is not None:
                query_article_pairs.extend(qa_pairs_in_sent)

    return query_article_pairs


parser = argparse.ArgumentParser()

parser.add_argument(
    '--pmid2info_path',
    type=str,
    required=True
)
parser.add_argument(
    '--output_dir',
    type=str,
    required=True
)
parser.add_argument(
    '--extract_method',
    choices=('iterative', 'total'),
    required=True
)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

non_usable_reflist = []  # papers with references list but no usable ref-list i.e. no references with any kind of ref id
papers_wo_reflist = []  # papers without reference list
non_pm_id_citations = []  # papers that have a references that is not pubmed or doi
num_doi_refs = 0  # number of references that only have doi
num_pubmed_refs = 0  # number of references that have pubmed id
cit_wo_pubid_doiid = []  # number of references that do not have pub-id tag
successful_reflist = 0  # paper that could be parsed til finish
unsuccesful_papers = [] # paper file which couldnt be read

lines_for_json = []

# pmc_dir = '/vol/tmp/lethanhd/pmc/oa_comm/xml/PMC000xxxxxx'
# files = [os.path.join(pmc_dir, file) for file in os.listdir(pmc_dir)]

with open('random_file_samples_50k', 'r') as f:
    files = f.read().splitlines()

reference_time = 0
query_article_time = 0
qid2info_write_time = 0
train2jsonl_time = 0

successful_get_qa_pairs = 0
num_fully_parsed = 0

print("Start loading pmid2info dict")
start = time.time()
with open(os.path.join(args.pmid2info_path), 'r') as f:
    pmid2info = json.load(f)
end = time.time()
pmid2info_load = end-start
print("Finished loading pmid2info dict")

qid2info = {}
train2jsonl = []

num_qa_pairs_in_files = {}
for file in tqdm(files):
    t0 = time.time()
    try:
        ref_dict = get_reference_dict(file)
    except Exception as e:
        lines_for_json.append(file)
        lines_for_json.append('\nrefdicterror: ' + str(repr(e)))
        lines_for_json.append('\n\n')

        continue
    t1 = time.time()
    reference_time += t1 - t0

    if ref_dict is None or len(ref_dict) == 0:
        continue

    t3 = time.time()
    try:
        query_article_pairs = get_query_article_pairs(file, ref_dict, pmid2info, method=args.extract_method)
    except Exception as e:
        lines_for_json.append(file)
        lines_for_json.append('\npapererror: ' + str(repr(e)))
        lines_for_json.append('\n\n')

        continue
    t4 = time.time()
    successful_get_qa_pairs += 1

    query_article_time += t4 - t3

    if query_article_pairs is None or len(query_article_pairs) == 0:
        continue

    could_generate_qa_pair = False
    num_qa_pairs = 0
    for query, articles in query_article_pairs:
        if len(articles) == 0:
            continue

        could_generate_qa_pair = True
        num_qa_pairs += len(articles)

        next_query_id = len(qid2info)
        qid2info[next_query_id] = query

        for article in articles:
            train2jsonl.append({"qid": str(next_query_id), "pmid": str(article), "click": 1})

    if could_generate_qa_pair:
        num_fully_parsed += 1
        num_qa_pairs_in_files[file] = num_qa_pairs

t6 = time.time()
with open(os.path.join(args.output_dir, 'qid2info.json'), 'w') as f:
    json.dump(qid2info, f, ensure_ascii=False, indent=4)
t7 = time.time()
qid2info_write_time = t7 - t6

t9 = time.time()
with open(os.path.join(args.output_dir, 'train.jsonl'), 'w') as f:
    for entry in train2jsonl:
        json.dump(entry, f)
        f.write('\n')
t10 = time.time()
train2jsonl_time = t10 - t9

lines = ['method ' + str(args.extract_method),
         '\nnum qid2info ' + str(len(qid2info)),
         '\nnum train2jsonl ' + str(len(train2jsonl)),
         '\nnum successful_get_qa_pairs ' + str(successful_get_qa_pairs),
         '\nnum num_fully_parsed ' + str(num_fully_parsed),
         '\ntime pmid2info_size ' + str(len(pmid2info)),
         '\ntime pmid2info_load_time ' + str(pmid2info_load),
         '\ntime reference_time ' + str(reference_time),
         '\ntime query_article_time ' + str(query_article_time),
         '\ntime qid2info_write_time ' + str(qid2info_write_time),
         '\ntime train2jsonl_time ' + str(train2jsonl_time),
         '\nnum faulty papers ' + str(len(non_usable_reflist)),
         '\npapers_wo_reflist ' + str(len(papers_wo_reflist)),
         '\nnum_pm_refs ' + str(num_pubmed_refs),
         '\nnum_doi_refs ' + str(num_doi_refs),
         '\nnum_cit_wo_pubid ' + str(len(cit_wo_pubid_doiid)),
         '\nsuccessful_reflist ' + str(successful_reflist),
         '\nunsuccesful_papers ' + str(len(unsuccesful_papers))]

if len(non_usable_reflist) > 0:
    lines.append('\nfaulty papers ' + str([non_usable_reflist[x] for x in [0, len(non_usable_reflist) // 2, -1]]))
if len(papers_wo_reflist) > 0:
    lines.append('\npapers_wo_reflist ' + str([papers_wo_reflist[x] for x in [0, len(papers_wo_reflist) // 2, -1]]))
if len(cit_wo_pubid_doiid) > 0:
    lines.append('\nnum_cit_wo_pubid ' + str([cit_wo_pubid_doiid[x] for x in [0, len(cit_wo_pubid_doiid) // 2, -1]]))
if len(unsuccesful_papers) > 0:
    lines.append('\nunsuccesfull_papers' + str([unsuccesful_papers[x] for x in [0, len(unsuccesful_papers) // 2, -1]]))

with open(os.path.join(args.output_dir, 'fullrun_stats'), 'w') as f:
    f.writelines(lines)

with open(os.path.join(args.output_dir, 'parse_full_text'), 'w') as f:
    f.writelines(lines_for_json)

with open(os.path.join(args.output_dir, 'citation_in_paper_stats'), 'w') as f:
    json.dump(num_qa_pairs_in_files, f, indent=4)

