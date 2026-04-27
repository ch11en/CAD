import random
import os
import json
from torch.utils.data import Dataset
import torch
# from utils.generate_data import get_data, get_domain, domain_map


def get_dataset(tokenizer, type_path, args):
    return GDPdataset(tokenizer=tokenizer,
                      data_dir=args.dataset,
                      data_type=type_path,
                      max_len=args.max_seq_length,
                      task=args.task,
                      truncate=args.truncate)


def read_line_examples_from_file(data_path, silence=False):
    """
    Read data from file, each line is:
    {
        "sentence": "We have gone for dinner only a few times but the same great quality and service is given.",
        "quads": [
            [
                "service",
                "great",
                "service general",
                "positive"
            ],
        ],
        "ID": 0
    },

    Return List[List[word]], List[Tuple]
    """
    with open(data_path, 'r', encoding='UTF-8') as file:
        data = json.load(file)

        sent = [item['sentence'].split() for item in data]
        quad = [[quad for quad in item['quads']] for item in data]  # 改成元组

        # if line != '':
        #     words, tuples = line.split('####')
        #     sents.append(words.split())
        #     labels.append(eval(tuples))

    return sent, quad


def get_para_asqp_targets(sents, labels, truncated=False):
    """
    Obtain the target sentence under the paraphrase paradigm
    This replicates the ABSA-QUAD approach
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            # TRUNCATED
            if truncated == True:
                ac = ac.split("#")[0]

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return sents.copy(), targets


def replace_unk_tokens(string):
    replace_dict = {"`": "'", }
    for pstr in replace_dict:
        string = string.replace(pstr, replace_dict[pstr])
    return string


def get_transformed_io(data_dir, dataset, truncate=False):
    sents, labels = read_line_examples_from_file(data_path=os.path.join(data_dir, dataset))  # Format : sent #### labels

    inputs, targets = get_data(sents, labels, truncate)

    # inputs=原始句子, targets=["THE", at, "IS", ot, "|", ac,  "|", sp], labels是数据集后#### 的四元组,可能有多个label
    return inputs, targets, labels


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, max_len=256, data_path=None, truncate=False):
        # './data/rest16/train.txt'
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = f'../data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.task = task
        self.data_type = data_type
        self.inputs = []
        self.targets = []
        self.contrastive_labels = {}
        self.sentence_strings = []
        self.truncate = truncate
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def get_raw_labels(self):
        results = get_transformed_io(self.data_path, self.data_dir, self.task, self.data_type, self.truncate)
        return results

    def _build_examples(self):

        inputs, targets, _ = get_transformed_io(self.data_path, self.data_dir, self.task, self.data_type, self.truncate)
        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


class GDPdataset(ABSADataset):
    def __init__(self):
        super().__init__()
        self._build_examples()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        sentiment_label = torch.tensor(self.contrastive_labels['sentiment'][index])
        aspect_label = torch.tensor(self.contrastive_labels['aspect'][index])
        opinion_label = torch.tensor(self.contrastive_labels['opinion'][index])

        grid_matrix_label = self.grid_matrix_labels[index]
        category_mask_matrix = self.category_mask_matrixes[index]

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                'sentiment_labels': sentiment_label,
                'opinion_labels': opinion_label,
                'aspect_labels': aspect_label,
                'grid_matrix_label': grid_matrix_label,
                'category_mask_matrix': category_mask_matrix
                }

    def _build_examples(self):

        inputs, targets, labels = get_transformed_io(self.data_path, self.data_dir, self.task, self.data_type,
                                                     self.truncate)

        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings

            input = ' '.join(inputs[i])
            input = replace_unk_tokens(input)
            target = targets[i]
            if isinstance(targets[i], list):
                target = " ".join(targets[i])
                target = replace_unk_tokens(target)

            tokenized_input = self.tokenizer.batch_encode_plus([input],
                                                               max_length=self.max_len,
                                                               padding="max_length",
                                                               truncation=True,
                                                               return_tensors="pt"
                                                               )
            # todo 2023 0204
            tokenized_target = self.tokenizer.batch_encode_plus([target],
                                                                max_length=self.max_len,
                                                                padding="max_length",
                                                                truncation=True,
                                                                return_tensors="pt"
                                                                )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        def get_sentiment_labels(labels_in):
            sentiment_dict = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
            sentiment_labels = []
            for ex in labels_in:
                label = list(set([quad[2] for quad in ex]))
                if len(label) == 1:
                    label = sentiment_dict[label[0]]
                else:
                    label = sentiment_dict['mixed']
                assert label in [0, 1, 2, 3]
                sentiment_labels.append(label)
            from collections import Counter
            print(f"Sentiment distribution: {Counter(sentiment_labels)}")
            return sentiment_labels

        def get_opinion_labels(labels_in):
            opinion_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
            opinion_labels = []
            for ex in labels_in:
                opinions = set([quad[3] for quad in ex])

                if 'NULL' not in opinions:
                    label = opinion_dict['EXPLICIT']
                else:
                    if len(opinions) == 1:
                        label = opinion_dict['NULL']
                    else:
                        label = opinion_dict['BOTH']

                opinion_labels.append(label)
            return opinion_labels

        def get_aspect_labels(labels_in):
            aspect_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
            aspect_labels = []
            for ex in labels_in:
                aspects = set([quad[0] for quad in ex])

                if 'NULL' not in aspects:
                    label = aspect_dict['EXPLICIT']
                else:
                    if len(aspects) == 1:
                        label = aspect_dict['NULL']
                    else:
                        label = aspect_dict['BOTH']

                aspect_labels.append(label)
            return aspect_labels

        def get_grid_labels(inputs_in, targets_in, labels_in, tokenizer, truncated=False):
            grid_member_dict = {
                # 'pad': -100,
                'pad': 32100,
                'none': 0,
                'aspect': 1,
                'opinion': 2,
                'negative': 3,
                'neutral': 4,
                'positive': 5,
            }
            grid_matrixes = []
            category_mask_matrixes = []
            for inidx in range(len(inputs_in)):
                sentence_encoded = inputs_in[inidx]["input_ids"].squeeze()
                targets_encoded = targets_in[inidx]["input_ids"].squeeze()
                non_pad_sentence_len = sum(tokenizer.pad_token_id != sentence_encoded).item()
                non_pad_target_len = sum(tokenizer.pad_token_id != targets_encoded).item()
                grid_matrix = torch.full((len(targets_encoded), len(sentence_encoded)),
                                         grid_member_dict['pad'])
                category_mask_matrix = torch.full((len(targets_encoded),), 0)
                # grid_matrix[:non_pad_target_len, :non_pad_sentence_len] = grid_member_dict['none']

                sentence_dense_to_decoded = []
                sentence_dense_str = ""
                last_sentence_dense_str_len = len(sentence_dense_str)
                for seidx in range(1, non_pad_sentence_len + 1):
                    sentence_dense_str = tokenizer.decode(sentence_encoded[:seidx],
                                                          clean_up_tokenization_spaces=False).replace(
                        ' ', '')
                    sentence_dense_to_decoded.extend(
                        [seidx - 1] * (len(sentence_dense_str) - last_sentence_dense_str_len))
                    last_sentence_dense_str_len = len(sentence_dense_str)

                target_dense_to_decoded = []
                target_dense_str = ""
                last_target_dense_str_len = len(target_dense_str)
                for seidx in range(1, non_pad_target_len + 1):
                    target_dense_str = tokenizer.decode(targets_encoded[:seidx],
                                                        clean_up_tokenization_spaces=False).replace(
                        ' ', '')
                    target_dense_to_decoded.extend(
                        [seidx - 1] * (len(target_dense_str) - last_target_dense_str_len))
                    last_target_dense_str_len = len(target_dense_str)
                aspects = list([quad[0] for quad in labels_in[inidx]])
                opinions = list([quad[3] for quad in labels_in[inidx]])
                categorys = list([quad[1] for quad in labels_in[inidx]])

                assert len(aspects) == len(opinions) == len(categorys)
                for asp, opn, raw_ac in zip(aspects, opinions, categorys):
                    if truncated == True:
                        raw_ac = raw_ac.split("#")[0]
                    domain = get_domain(raw_ac)
                    asc = domain_map[domain].get(raw_ac, raw_ac)

                    asp_dense, opn_dense, asc_dense = asp.replace(" ", ""), opn.replace(" ", ""), asc.replace(" ", "")

                    asp_dense = replace_unk_tokens(asp_dense)
                    opn_dense = replace_unk_tokens(opn_dense)
                    asc_dense = replace_unk_tokens(asc_dense)

                    if asc_dense not in target_dense_str:
                        print(asc_dense, target_dense_str)
                        print("inidx", inidx)
                    assert asc_dense in target_dense_str

                    asc_left = target_dense_str.index(asc_dense)
                    asc_right = asc_left + len(asc_dense) - 1

                    asc_left, asc_right = target_dense_to_decoded[asc_left], target_dense_to_decoded[asc_right] + 1

                    grid_matrix[asc_left: asc_right, :non_pad_sentence_len] = grid_member_dict['none']

                    category_mask_matrix[asc_left: asc_right] = 1

                    if asp_dense != "NULL":
                        if asp_dense in sentence_dense_str:
                            assert asp_dense in sentence_dense_str
                            asp_left = sentence_dense_str.index(asp_dense)
                            asp_right = asp_left + len(asp_dense) - 1
                            asp_left, asp_right = sentence_dense_to_decoded[asp_left], sentence_dense_to_decoded[
                                                                                           asp_right] + 1
                            # grid_matrix[asc_left: asc_right, :non_pad_sentence_len] = grid_member_dict['none']
                            grid_matrix[asc_left: asc_right, asp_left: asp_right] = grid_member_dict['aspect']
                            # grid_matrix[asp_left: asp_right, asp_left: asp_right] = grid_member_dict['aspect']  # asp在句子中的位子置1
                        else:
                            print(asp_dense, sentence_dense_str)
                    if opn_dense != "NULL":
                        if opn_dense in sentence_dense_str:
                            assert opn_dense in sentence_dense_str
                            opn_left = sentence_dense_str.index(opn_dense)
                            opn_right = opn_left + len(opn_dense) - 1
                            opn_left, opn_right = sentence_dense_to_decoded[opn_left], sentence_dense_to_decoded[
                                                                                           opn_right] + 1
                            # grid_matrix[asc_left: asc_right, :non_pad_sentence_len] = grid_member_dict['none']
                            grid_matrix[asc_left: asc_right, opn_left: opn_right] = grid_member_dict['opinion']
                            # grid_matrix[opn_left: opn_right, opn_left: opn_right] = grid_member_dict['opinion']  # opn在句子中的位子置2
                        else:
                            print(opn_dense, sentence_dense_str)

                    # if asp_dense != "NULL" and opn_dense != "NULL":
                    #     grid_matrix[asp_left: asp_right, opn_left: opn_right] = grid_member_dict['negative']
                    #     grid_matrix[opn_left: opn_right, asp_left: asp_right] = grid_member_dict['negative']
                grid_matrixes.append(grid_matrix)
                category_mask_matrixes.append(category_mask_matrix)
            from collections import Counter
            print("Sentiment distribution")
            # print(Counter(grid_matrixes))
            return grid_matrixes, category_mask_matrixes

        self.contrastive_labels['sentiment'] = get_sentiment_labels(labels)
        self.contrastive_labels['opinion'] = get_opinion_labels(labels)
        self.contrastive_labels['aspect'] = get_aspect_labels(labels)
        self.grid_matrix_labels, self.category_mask_matrixes = get_grid_labels(self.inputs,
                                                                               self.targets,
                                                                               labels,
                                                                               self.tokenizer,
                                                                               self.truncate)
