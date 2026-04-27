import numpy as np


def load_mappings():
    """
    Load category mappings used to map existing labelset to human-readable variant
    """
    import os
    import json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'category_mappings.json')) as ofile:
        data_json = json.load(ofile)
    return data_json


mappings = load_mappings()

use_the_gpu = False

sent_to_word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
sent_to_opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sent_to_word_to_opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

lap_map_parent = mappings['lap_map_parent']
lap_map = mappings['lap_map']
res_map = mappings['res_map']

res_map_dict = {}
for res in res_map:  # ex: elt: ['the location', 'LOCATION#GENERAL']
    res_map_dict[res[0]] = res[1]

lap_map_dict = {}
for lap in lap_map:
    lap_map_dict[lap[0]] = lap[1]

lap_parent_map_dict = {}
for lap_p in lap_map_parent:
    lap_parent_map_dict[lap_p[0]] = lap_p[1]

domain_map = {'restaurant': res_map_dict,
              'laptop': lap_map_dict,
              'laptop_parent': lap_parent_map_dict}


def get_domain(label):
    for key in domain_map:
        if label in domain_map[key]:
            print(f'key: {key}')
            return key
    assert False, "invalid domain"


def ex_contains_implicit_opinion(quads):
    return any([quad[3] == 'NULL' and quad[0] != 'NULL' for quad in quads])


def ex_contains_implicit_aspect(quads):
    return any([quad[0] == 'NULL' and quad[3] != 'NULL' for quad in quads])


def ex_contains_full_implicit(quads):
    return any([quad[0] == 'NULL' and quad[3] == 'NULL' for quad in quads])


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))
    return results


def get_pos_vec_bert(quad, sent):
    """
    returns binary vector indicating tokens relevant to current quadruple
    1 indicates token is part of either this quads 1) opinion or 2) aspect span
    0 indicates it is not
    """
    current_spans = {quad[0], quad[3]}  # aspect and opinion terms
    zeroes_vec = np.zeros(len(sent))
    for ent in current_spans:
        ent_list = ent.split(" ")
        # find locations of explicit terms to focus on
        curr_indices = find_sub_list(ent_list, sent)
        if curr_indices:
            first_result = curr_indices[0]
            for idx in range(first_result[0], first_result[1]):
                zeroes_vec[idx] = 1
    return zeroes_vec


def get_data(sents, labels, truncated=False):
    def inner_fn(sent, label):
        ''' Handles a single review : get aspects'''
        aspect_terms = set()
        for quad in label:
            aspect_terms.update([quad[0], quad[3]])
        utt_str = " ".join(sent)
        indices = {}
        for term in aspect_terms:
            if term != 'NULL':
                indices[term] = utt_str.find(term)
            else:
                indices[term] = len(utt_str) + 1

        sorted_aspects = sorted(aspect_terms, key=indices.get)

        outputs = []
        seen_aspects = set()
        covered = set()

        for aspect in sorted_aspects:
            seen_aspects.add(aspect)
            for quad in label:
                # if we can produce the quad using the seen aspects, then generate the summary
                if quad[0] in seen_aspects and quad[3] in seen_aspects and tuple(quad) not in covered:
                    if len(quad) == 4:
                        at, raw_ac, sp, ot = quad
                    # Use this just for the truncated labelsets dataset
                    if truncated == True:
                        raw_ac = raw_ac.split("#")[0]

                    domain = get_domain(raw_ac)
                    ac = domain_map[domain].get(raw_ac, raw_ac)

                    covered.add(tuple(quad))
                    man_ot = sent_to_word_to_opinion[sp]  # 'POS' -> 'good'
                    if at == 'NULL':  # for implicit aspect term
                        at = 'it'

                    revised_quad = [at, ac, man_ot, ot]

                    positon_vecor = get_pos_vec_bert(revised_quad, sent)
                    # template = ["THE", at, "IS", ot, "|", ac,  "|", sp]
                    one_quad_sent = ["THE", at, "IS", ot, "|", ac, "|", sp]

                    outputs.append((one_quad_sent, positon_vecor))

        sent_len = len(sent)

        sorted_outputs = sorted(outputs, key=lambda x: (
        max(loc for loc, val in enumerate(x[1]) if val == 1) if 1 in x[1] else sent_len, x[0]))

        total_sent = []
        total_pos = None
        total_neg = None

        for idx, output in enumerate(sorted_outputs):
            total_sent += output[0]

            if idx != len(sorted_outputs) - 1:
                # add SSEP token
                total_sent.append('[SSEP]')

        return sent.copy(), total_sent

    sents, outputs = list(zip(*[inner_fn(sents[idx], labels[idx]) for idx in range(len(sents))]))  # modify

    return sents, outputs
