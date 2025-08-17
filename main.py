import os
import time
import random
import numpy as np
import torch
import lightning as pl
from transformers.models.gpt_neox.modeling_gpt_neox import attention_mask_func

pl.seed_everything(42)
from lightning.pytorch.callbacks import BasePredictionWriter

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from utils import params_count, load_json, tokenize, load_line_json, save_line_json, save_json, tgenerate_batch, \
    simple_text_len, auto_init
from utils.quad import make_quads_seq, parse_quads_seq, get_quad_aspect_opinion_num

from model import *
from pytorch_lightning.callbacks import TQDMProgressBar
import argparse
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataModule(pl.LightningDataModule):
    def __init__(self, args):

        super().__init__()
        self.args = args
        self.data_dir = args.data_dir if args.dataset == '' else os.path.join(args.data_dir, args.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, legacy=True)
        self.raw_datasets = {}
        self.new_datasets = {}

    def load_labeled_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name = os.path.join(self.data_dir, 'dev.json')
        test_file_name = os.path.join(self.data_dir, 'test.json')

        train_examples = load_json(train_file_name)
        dev_examples = load_json(dev_file_name)
        test_examples = load_json(test_file_name)

        self.raw_datasets = {
            'train': train_examples,  # 含以下键值对['sentence', 'quads', 'ID']
            'dev': dev_examples,
            'test': test_examples}

        self.raw_datasets_with_target_sent = self.add_target_data_to_raw_dataset(
            self.raw_datasets)  # 含以下键值对['sentence', 'quads', 'ID', 'target_sentence']
        self.new_datasets = self.add_tokenized_input_to_raw_dataset(
            self.raw_datasets_with_target_sent)  # [ {'sentence', 'quads', 'ID', 'target_sentence', 'tokenized_sent_input_ids', 'tokenized_sent_attention_mask', 'tokenized_target_sent'} ]

        if args.self_training_data_dir and args.filter_setting != 'none':
            self.add_self_training_data()

        print('-----------data statistic-------------')
        for mode in ('train', 'dev', 'test'):
            num_sentences = len(self.raw_datasets[mode])
            num_quads = sum([get_quad_aspect_opinion_num(example)[0] for example in self.raw_datasets[mode]])
            print(f'{mode.upper():<5} | Sentences: {num_sentences:<5} | Quad: {num_quads:<5}')
        print('--------------------------------------')

    def generate_target_sentence(self, quad):
        at, ot, ac, sp = quad  # 'train', 'dev', 'test'
        if at == 'NULL':
            at = 'none'
        if ot == 'NULL':
            ot = 'none'
        quad_seq = ' | '.join([at, ot, ac,
                               sp])  # after : [aspect | opinion | category | sentiment]   ;   original : [category | sentiment | aspect | opinion]
        # quad_seq = ["THE", at, "IS", ot, "|", ac, "|", sp]
        return [quad_seq]

    def add_target_data_to_raw_dataset(self, raw_datasets):
        for each_mode_raw_datasets in raw_datasets:
            for item in raw_datasets[each_mode_raw_datasets]:
                item["target_sentence"] = [self.generate_target_sentence(quad) for quad in item["quads"]]
        return raw_datasets

    def add_tokenized_input_to_raw_dataset(self, raw_datasets_with_target_sent):
        for each_mode_raw_datasets in raw_datasets_with_target_sent:  # 遍历 'train', 'dev', 'test'
            for item in raw_datasets_with_target_sent[each_mode_raw_datasets]:  # 遍历每个数据项
                # print('sentence : ', item['sentence'])

                tokenized_input = tokenizer.batch_encode_plus(  # 对 target_sentence 进行编码
                    [item['sentence']],  # 输入文本（列表形式，支持批量）
                    max_length=128,  # 最大长度（超出截断）
                    padding="max_length",  # 填充到 max_length
                    truncation=True,  # 允许截断
                    return_tensors="pt"  # 返回Pytorch张量
                )

                item['tokenized_sent_input_ids'] = tokenized_input['input_ids']  # 将编码结果存入字典
                item['tokenized_sent_attention_mask'] = tokenized_input['attention_mask']
        # print(raw_datasets_with_target_sent['each_mode_raw_datasets']['tokenized_sent_input_ids'])
        for each_mode_raw_datasets in raw_datasets_with_target_sent:
            for item in raw_datasets_with_target_sent[each_mode_raw_datasets]:
                for ts in item['target_sentence']:
                    tokenized_quads = []  # 存储每个quad的tokenized结果

                    tokenized_target = tokenizer.batch_encode_plus(
                        ts,
                        max_length=128,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    tokenized_quads.append(tokenized_target["input_ids"])
                    tokenized_quads.append(tokenized_target["attention_mask"])

                    item['tokenized_target_sent'] = tokenized_quads

        return raw_datasets_with_target_sent

    def add_self_training_data(self):
        setting, k = args.filter_setting.split('_')
        k = int(k)

        print(f'setting: {setting} | k: {k}')
        try:
            self_training_data = load_json(args.self_training_data_dir)
        except:
            self_training_data = list(load_line_json(args.self_training_data_dir))

        if len(self_training_data) >= 110_000:  # 数据截断
            self_training_data = self_training_data[10_000:110_000]  # 如果数据量超过 110,000 条，则截取第 10,000 到 110,000 条数据。

        self_training_data = [{
            'ID': example['ID'],
            'sentence': example['sentence'],
            'quads_seq': example['quad_preds'][0],
            'reward': example['reward'] if 'reward' in example else [None],
            'quad_preds': example['quad_preds'],
        } for example in self_training_data]

        if setting == 'full':  # 如果 setting 是 full，则不进行过滤
            pass

        else:  # 否则，setting 应该是形如 start-end 的字符串，表示按 reward 排序后截取数据的百分比范围
            try:
                start, end = setting.split('-')
                start = int(start) / 100 * len(self_training_data)
                end = int(end) / 100 * len(self_training_data)

                self_training_data = sorted(self_training_data, key=lambda e: e['reward'][0], reverse=True)[
                                     int(start): int(end)]  # 例如，setting 为 10-90 时，表示截取 reward 最高的前 10% 到 90% 的数据。

            except:
                raise NotImplementedError(f'Unknown setting: {args.filter_setting}')

        # 随机采样
        if k > 0:
            random.seed(args.seed)
            self_training_data = random.sample(self_training_data, k=k)  # 如果 k 大于 0，则从过滤后的数据中随机采样 k 条数据
            mean_reward = sum(example['reward'][0] for example in self_training_data) / len(self_training_data)
            print(f'mean_reward: {mean_reward}')
            self.raw_datasets['train'] += self_training_data  # 将采样后的数据添加到 self.raw_datasets['train'] 中

    def load_unlabeled_dataset(self, max_example_num=1_000_000):
        import spacy
        import re

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('sentencizer')

        min_length = 5
        max_length = 100
        if self.args.mode == 'predict':
            # print('self.args.mode == predict')
            self.data_dir = "data/raw/yelp/100k_1.json"

        dataset = list(load_line_json(self.data_dir))

        # print(f"Loaded {len(dataset)} examples from {self.data_dir}")  # 添加这行

        def process_and_filter(raw_sentence):
            raw_sentence = str(raw_sentence).strip()
            raw_sentence = raw_sentence.replace('\r', '')  # '(good)' -> '( good )'
            new_sentence = re.sub(r'\((?P<v1>[^ ])(?P<v2>.*)(?P<v3>[^ ])\)',
                                  lambda x: '( ' + x.group('v1') + x.group('v2') + x.group('v3') + ' )',
                                  raw_sentence)  # 匹配括号内的内容，要求括号内的第一个和最后一个字符是非空格字符。在括号内的内容前后添加空格，使其格式变为 ( content )。

            if not (min_length <= simple_text_len(new_sentence) <= max_length):
                return None

            return new_sentence

        predict_examples = []
        for batch_examples in tgenerate_batch(dataset, bz=32):

            texts = [example['Text'] for example in batch_examples]
            docs = nlp.pipe(texts, disable=['tagger', 'tok2vec', 'parser', 'lemmatizer', 'ner'])

            for doc, example in zip(docs, batch_examples):
                for i, sentence in enumerate(doc.sents):
                    if (sentence := process_and_filter(sentence)) is not None:
                        new_example = {
                            'ID': f"{example['ID']}-{i}",
                            'sentence': sentence,
                            'full_review': example['Text']
                        }
                        predict_examples.append(new_example)  # 加入新样本

                if max_example_num > 0 and len(predict_examples) >= max_example_num:
                    break

        self.new_datasets = {
            'predict': predict_examples}  # original : self.raw_datasets = {'predict': predict_examples}

        print('-----------data statistic-------------')
        # print('Predict', len(self.new_datasets['predict'])) # original : print('Predict', len(self.raw_datasets['predict']))

    def prepare_data(self):  # 会被lighting自动调用
        if args.mode == 'train_test':
            self.load_labeled_dataset()

        elif args.mode == 'predict':
            self.load_unlabeled_dataset()

    def get_dataloader(self, mode, batch_size, shuffle):

        # print(f'self.new_datasets: {self.new_datasets.keys()}')  # self.new_datasets: dict_keys(['train', 'dev', 'test'])
        dataloader = DataLoader(
            # dataset=self.raw_datasets[mode],
            dataset=self.new_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,
            prefetch_factor=2,
            num_workers=1,
            persistent_workers=True,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer,
                max_seq_length=args.max_seq_length,
                mode=mode,
                dataset=args.dataset
            ),
            drop_last=True,
        )

        # print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', args.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", args.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", args.eval_batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.get_dataloader("predict", args.eval_batch_size, shuffle=False)


class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode, dataset):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.dataset = dataset

    def tok(self, text, max_seq_length):
        return tokenize(self.tokenizer, text, max_seq_length)

    def __call__(self, examples):
        # print(f'examples: {examples}') # list

        # Filter out empty examples
        examples = [ex for ex in examples if len(ex['sentence']) > 0]
        if not examples:
            print("WARNING: All examples were filtered out in DataCollator!")
            return {
                'input_ids': torch.empty(0, dtype=torch.long),
                'attention_mask': torch.empty(0, dtype=torch.long),
                'examples': []
            }

        text = [example['sentence'] for example in examples]
        batch_encodings = self.tok(text, -1)

        all_quads = []

        if self.mode != 'predict':
            # print(f'examples: {examples[0].keys()}')
            all_quads = []
            for item in examples:
                quads = item.get('quads', [])  # 检查是否有 'quads' 字段，没有则提供默认值（如空列表）
                all_quads.append(quads)

        # print(f'all_quads: {all_quads}')
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        target_ids = None
        target_seqs = None
        target_mask = None
        ac_grid_label = None
        ac_grid_mask = None

        contrastive_labels = {}

        if self.mode in ('train', 'dev', 'test'):
            target_seqs, target_ids, target_mask = self.make_labels(
                examples)  # target_seqs : [category | sentiment | aspect | opinion]

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            if target_ids is not None:
                target_ids = target_ids[:, :self.max_seq_length]

        if self.mode != 'predict':
            contrastive_labels['at'] = self.get_at_labels(
                all_quads)  # example : self.contrastive_labels: {'at': [1, 1, 1, 1, 1, 1, 1, 1], 'ot': [1, 1, 1, 1, 1, 1, 1, 1], 'sp': [2, 2, 2, 2, 2, 2, 2, 2]}
            contrastive_labels['ot'] = self.get_ot_labels(all_quads)
            contrastive_labels['sp'] = self.get_sp_labels(all_quads)

            ac_grid_label, ac_grid_mask = self.get_grid_labels(input_ids, target_ids, target_seqs)

        if self.mode in ('train', 'dev', 'test'):
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': target_ids,
                'target_mask': target_mask,
                'examples': examples,
                'ac_grid_label': ac_grid_label,
                'ac_grid_mask': ac_grid_mask,
                'contrastive_labels': contrastive_labels,
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': target_ids,
                'target_mask': target_mask,
                'examples': examples,
            }

    def make_labels(self, examples):
        target_seqs = [self.make_quads_seq(example) for example in
                       examples]  # [category | sentiment | aspect | opinion]

        batch_encodings = self.tok(target_seqs, -1)

        target_ids = batch_encodings['input_ids']

        target_ids = torch.tensor([[(l if l != self.tokenizer.pad_token_id else -100)
                                    for l in label]
                                   for label in target_ids])

        target_mask = batch_encodings['attention_mask']
        return target_seqs, target_ids, target_mask

    def make_quads_seq(self, example):
        if 'quads_seq' in example:
            return example['quads_seq']

        return make_quads_seq(example)  # [category | sentiment | aspect | opinion]

    def replace_unk_tokens(self, string):
        replace_dict = {"`": "'", }
        for pstr in replace_dict:
            string = string.replace(pstr, replace_dict[pstr])
        return string

    def get_element_label(self, targets_seqs):
        '''
        example = ['food quality | positive | none | delicious',
                   'food quality | positive | food | good ; food quality | neutral | food | not outstanding',
                   'restaurant general | positive | none | best',
                   'drinks quality | positive | seasonal beer | none',
                   'location general | positive | view of river and nyc | nice', 'food quality | negative | dishes | average at best ; food prices | negative | dishes | expensive',
                   'ambience general | positive | none | tranquility',
                   'ambience general | positive | spot | unpretentious ; food quality | positive | sushi | good ; service general | positive | service | pleasant ; service general | positive | service | effective ; service general | positive | service | unassuming',
                   'food quality | negative | food | aweful',
                   'ambience general | positive | restaurant | family feel ; food style_options | positive | portions | enormous ; food style_options | positive | veal | none',
                   'drinks style_options | positive | drink menu | love', 'food quality | positive | none | great ; food quality | positive | none | original',
                   'food prices | positive | none | extremely well',
                   'food quality | negative | meal | inedible ; restaurant prices | negative | none | none',
                   'restaurant general | positive | none | none', 'food quality | positive | seafood dynamite | otherworldly']
        '''

        all_aspects = []
        all_opinions = []
        all_categorys = []
        # print(f'targets_seqs: {targets_seqs}')

        for sentence in targets_seqs:
            # 分割句子中的多个 quad（用 ; 分隔）
            quads = [quad.strip() for quad in sentence.split(';')]
            # print(f'quads: {quads}')
            # 初始化当前句子的 aspects、opinions 和 categorys
            aspects = []
            opinions = []
            categorys = []

            for quad in quads:
                # 分割 quad 的四个部分（用 | 分隔）
                parts = [part.strip() for part in quad.split('|')]
                if len(parts) == 4:  # 确保是有效的 quad
                    category, sentiment, aspect, opinion = parts
                    aspects.append(aspect)
                    opinions.append(opinion)
                    categorys.append(category)

            all_aspects.append(aspects)
            all_opinions.append(opinions)
            all_categorys.append(categorys)
            # print(f'at : {all_aspects}')
            # print(f'op : {all_opinions}')
            # print(f'ac : {all_categorys}')
        return all_aspects, all_opinions, all_categorys  # list, contain each element label

    def get_at_labels(self, labels):
        at_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        at_labels = []
        # print(f'labels: {labels}')
        for label in labels:

            at = set([quad[0] for quad in label])

            # print(f'at label: {at}')
            if 'NULL' not in at:
                at = at_dict['EXPLICIT']
            else:
                if len(at) == 1:
                    at = at_dict['NULL']
                else:
                    at = at_dict['BOTH']

            at_labels.append(at)
        return at_labels

    def get_ot_labels(self, labels):
        ot_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        ot_labels = []

        for label in labels:
            ot = set([quad[1] for quad in label])

            if 'NULL' not in ot:
                ot = ot_dict['EXPLICIT']
            else:
                if len(ot) == 1:
                    ot = ot_dict['NULL']
                else:
                    ot = ot_dict['BOTH']

            ot_labels.append(ot)

        return ot_labels

    def get_sp_labels(self, labels):
        sp_dict = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
        sp_labels = []

        for label in labels:

            sp = list(set([quad[3] for quad in label]))
            if len(sp) == 1:
                sp = sp_dict[sp[0]]
            else:
                sp = sp_dict['mixed']
            assert sp in [0, 1, 2, 3]
            sp_labels.append(sp)

        # from collections import Counter
        # print("Sentiment distribution")
        # print(Counter(sentiment_labels))

        return sp_labels

    def get_grid_labels(self, inputs_in, targets_in, target_seqs, truncated=False):

        grid_member_dict = {'pad': -100, 'none': 0, 'aspect': 1, 'opinion': 2, 'negative': 3, 'neutral': 4,
                            'positive': 5, }
        grid_matrixes = []
        mask_matrixes = []

        for input_idx in range(len(inputs_in)):

            inputs_encoded = inputs_in[input_idx].squeeze()
            targets_encoded = targets_in[input_idx].squeeze()
            # print(f'target_seqs: {target_seqs}')

            non_pad_sentence_len = sum(self.tokenizer.pad_token_id != inputs_encoded).item()
            non_pad_target_len = sum(self.tokenizer.pad_token_id != targets_encoded).item()

            grid_matrix = torch.full((len(targets_encoded), len(inputs_encoded)), grid_member_dict['pad'])
            ac_grid_mask = torch.full((len(targets_encoded),), 0)
            # grid_matrix[:non_pad_target_len, :non_pad_sentence_len] = grid_member_dict['none']

            sentence_dense_to_decoded = []
            sentence_dense_str = ""
            last_sentence_dense_str_len = len(sentence_dense_str)

            # 填充
            for i in range(1, non_pad_sentence_len + 1):
                sentence_dense_str = tokenizer.decode(inputs_encoded[:i], clean_up_tokenization_spaces=False).replace(
                    ' ', '')
                sentence_dense_to_decoded.extend([i - 1] * (len(sentence_dense_str) - last_sentence_dense_str_len))
                last_sentence_dense_str_len = len(sentence_dense_str)

            target_dense_to_decoded = []
            target_dense_str = ""
            last_target_dense_str_len = len(target_dense_str)

            for i in range(1, non_pad_target_len + 1):
                targets_encoded[targets_encoded == -100] = 0
                target_dense_str = tokenizer.decode(targets_encoded[:i], ignore=-100,
                                                    clean_up_tokenization_spaces=False).replace(' ', '')

                target_dense_to_decoded.extend([i - 1] * (len(target_dense_str) - last_target_dense_str_len
                                                          ))
                last_target_dense_str_len = len(target_dense_str)
            ats, ops, acs = self.get_element_label(target_seqs)

            assert len(ats) == len(ops) == len(acs)
            for at_dense, ot_dense, ac_dense in zip(ats, ops, acs):

                for i in range(len(at_dense)):
                    at_dense[i] = at_dense[i].replace(" ", "")
                    if at_dense != 'none':
                        if at_dense[i] in sentence_dense_str:
                            assert at_dense[i] in sentence_dense_str
                            # 确定at元素左右位置
                            at_dense_left = sentence_dense_str.index(at_dense[i])
                            at_dense_right = at_dense_left + len(at_dense[i]) - 1

                            at_dense_left, at_dense_right = sentence_dense_to_decoded[at_dense_left], \
                            sentence_dense_to_decoded[at_dense_right] + 1
                            # grid_matrix[asc_left: asc_right, :non_pad_sentence_len] = grid_member_dict['none']
                            # grid_matrix[asc_left: asc_right, at_dense_left: at_dense_right] = grid_member_dict['aspect']
                            grid_matrix[at_dense_left: at_dense_right, at_dense_left: at_dense_right] = \
                            grid_member_dict['aspect']  # at_dense在句子中的位子置1
                        # else:
                        #     print(f'at : {at_dense[i]} is not in sentence_dense_str: {sentence_dense_str}')

                for i in range(len(ot_dense)):
                    ot_dense[i] = ot_dense[i].replace(" ", "")
                    if ot_dense != 'none':
                        if ot_dense[i] in sentence_dense_str:
                            assert ot_dense[i] in sentence_dense_str
                            # 确定ot元素左右位置
                            op_dense_left = sentence_dense_str.index(ot_dense[i])
                            op_dense_right = op_dense_left + len(ot_dense[i]) - 1

                            op_dense_left, op_dense_right = sentence_dense_to_decoded[op_dense_left], \
                            sentence_dense_to_decoded[op_dense_right] + 1
                            # grid_matrix[asc_left: asc_right, :non_pad_sentence_len] = grid_member_dict['none']
                            # grid_matrix[asc_left: asc_right, op_left: op_right] = grid_member_dict['opinion']
                            grid_matrix[op_dense_left: op_dense_right, op_dense_left: op_dense_right] = \
                            grid_member_dict['opinion']  # op在句子中的位子置2
                        # else:
                        #     print(f'op : {ot_dense} is not in sentence_dense_str: {sentence_dense_str}')

                for i in range(len(ac_dense)):
                    ac_dense[i] = ac_dense[i].replace(" ", "")
                    if ac_dense[i] in target_dense_str:
                        assert ac_dense[i] in target_dense_str
                        ac_dense_left = target_dense_str.index(ac_dense[i])
                        ac_dense_right = ac_dense_left + len(ac_dense[i]) - 1
                        ac_dense_left, ac_dense_right = target_dense_to_decoded[ac_dense_left], target_dense_to_decoded[
                                                                                                    ac_dense_right] + 1
                        grid_matrix[ac_dense_left: ac_dense_right, :non_pad_sentence_len] = grid_member_dict['none']
                        ac_grid_mask[ac_dense_left: ac_dense_right] = 1
                    else:
                        pass
                        # print(f'ac : {ac_dense} is not in target_dense_str: {target_dense_str}')

                # ac_dense[i] = ac_dense[i].replace(" ", "")
                # if ac_dense[i] not in target_dense_str:
                #     print(f'ac : {ac_dense[i]} is not in target_dense_strs: {target_dense_str}')
                #     assert ac_dense[i] in target_dense_str
                # # 确定ot元素左右位置
                # ac_dense_left = target_dense_str.index(ac_dense[i])
                # ac_dense_right = ac_dense_left + len(ac_dense[i]) - 1
                # ac_dense_left, ac_dense_right = target_dense_to_decoded[ac_dense_left], target_dense_to_decoded[ac_dense_right] + 1
                # grid_matrix[ac_dense_left: ac_dense_right, :non_pad_sentence_len] = grid_member_dict['none']
                # ac_grid_mask[ac_dense_left: ac_dense_right] = 1

                # if at != "NULL" and opn_dense != "NULL":
                #     grid_matrix[at_left: at_right, opn_left: opn_right] = grid_member_dict['negative']
                #     grid_matrix[opn_left: opn_right, at_left: at_right] = grid_member_dict['negative']

            # print(f'grid_matrixes: {grid_matrixes}')

            grid_matrixes.append(grid_matrix)
            mask_matrixes.append(ac_grid_mask)

        return grid_matrixes, mask_matrixes


class CADModule(pl.LightningModule):
    def __init__(self, args):
        super(LightningModule_SL, self).__init__()
        self.args = args
        self.ac_loss_lambda = 0.2
        self.sp_loss_lambda = 0.2
        self.grid_loss_lambda = 0.4

        self.grid_learner = Grid_learner(self.args)
        self.at_learner = Element_learner(self.args.model_name_or_path)
        self.ac_learner = Element_learner(self.args.model_name_or_path)
        self.ot_learner = Element_learner(self.args.model_name_or_path)
        self.sp_learner = Element_learner(self.args.model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_result = 0
        self.vp_dropout_time = 5

        self.dropout = nn.Dropout(0.4)  # 0.4
        self.mc_dropout_num = self.args.mc_forward_num  # 5
        self.gama = 10
        self.m = 0.4

    def save_model(self, args):
        dir_name = os.path.join(args.output_dir, 'model', f'dataset={args.dataset},b={args.subname},seed={args.seed}')
        print('\n', f'save model to : {dir_name}')

        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def vague_matrix_generation(self, sequence_output):  # sequence_output = decoder_last_layer_hiddens
        '''多次抽样模型输出来获得更稳健的预测或不确定性估计'''
        all_logits = None

        for num in range(self.vp_dropout_time):  # 5
            temp = self.dropout(sequence_output)

            if self.model.config.tie_word_embeddings:
                temp = temp * (self.model.model_dim ** -0.5)

            lm_logits = self.model.lm_head(temp).unsqueeze(0)

            if all_logits is not None:
                all_logits = torch.cat((all_logits, lm_logits), dim=0)
            else:
                all_logits = lm_logits
        return all_logits

    def vague_samples_acquisition(self, all_logits, labels):
        all_c = self.MAX(all_logits)
        negative_samples_masks = self.get_vague_samples_masks(all_c, labels, all_logits)
        return negative_samples_masks

    @torch.no_grad()
    def MAX(self, all_logits, top_k=1):
        all_top_k_ids = torch.topk(all_logits, top_k, dim=-1)
        return all_top_k_ids.indices.squeeze(-1)

    @torch.no_grad()
    def get_vague_samples_masks(self, all_c, labels, all_logits):
        '''
            生成一系列负样本掩码，用于标记哪些样本是负样本
            all_c.size() = [5, 8, 128]
            labels.size() = [8, 128]
            all_logits.size() = [5, 8, 128, 32101]
        '''
        batch_size = labels.shape[0]  # 8
        all_lengths = (labels != 0).sum(-1)  # [8]  value : tensor([128, 128, 128, 128, 128, 128, 128, 128])
        mask_results = torch.zeros_like(all_logits)  # [5, 8, 128, 32101]

        updated_labels = labels
        '''modified to 32099'''
        old_value = -100
        new_value = 0

        # 复制原始列表以创建一个新的列表,替换原有的值
        updated_labels[updated_labels == old_value] = new_value

        for i in range(all_c.shape[0]):  # 5
            for j in range(batch_size):  # 8
                mask_results[i, j, :all_lengths[j]] = mask_results[i, j, :all_lengths[j]].scatter(-1, all_c[i, j,
                                                                                                      :all_lengths[
                                                                                                          j]].unsqueeze(
                    -1), 1)

            # The label position is filled with 0
            mask_results[i] = mask_results[i].scatter(-1, labels.unsqueeze(-1), 0)

        return mask_results

    def get_mul_loss(self, N_mask_results, lm_labels, softmax_logits, label_masks):
        mc_forward_num = softmax_logits.shape[0]
        mask_results = N_mask_results
        n_logits = (self.gama * softmax_logits).exp() * mask_results
        n_logits = n_logits.sum(0).sum(-1)

        # get positive samples
        labels = lm_labels.unsqueeze(0).repeat(mc_forward_num, 1, 1).unsqueeze(-1)
        p_logits = torch.gather((- (self.gama * softmax_logits)).exp(), -1, labels)
        p_logits = p_logits.sum(0).squeeze(-1)

        loss = torch.log(1 + math.exp(self.m * self.gama) * n_logits * p_logits) * label_masks
        return loss.sum()

    def get_mse_loss(self, all_log_softmax_logits, lm_labels):
        mc_forward_num = all_log_softmax_logits.shape[0]
        vocab_size = all_log_softmax_logits.shape[-1]
        loss_fct = nn.NLLLoss(ignore_index=-100, reduction='sum')

        all_likelihood_loss = None
        for i in range(mc_forward_num):
            log_softmax_logits = all_log_softmax_logits[i]
            likelihood_loss = loss_fct(log_softmax_logits.reshape(-1, vocab_size), lm_labels.view(-1))
            cur_loss = likelihood_loss

            if all_likelihood_loss is None:
                all_likelihood_loss = cur_loss.unsqueeze(0)
            else:
                all_likelihood_loss = torch.cat((all_likelihood_loss, cur_loss.unsqueeze(0)), dim=0)

        all_likelihood_loss = torch.mean(all_likelihood_loss, dim=0)
        return all_likelihood_loss

    def get_mi_loss(self, label_masks, softmax_logits, all_log_softmax_logits):
        regular_loss = softmax_logits * all_log_softmax_logits
        label_masks = label_masks.unsqueeze(0).unsqueeze(-1)
        regular_loss = (regular_loss * label_masks).sum()
        return -regular_loss

    def compute_loss(self, all_logits, N_mask_results, lm_labels):
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax_fct = nn.Softmax(dim=-1)

        label_masks = torch.ones_like(lm_labels)
        label_masks[lm_labels == 0] = 0

        softmax_logits = softmax_fct(all_logits)

        all_log_softmax_logits = log_softmax(all_logits)

        mul_loss = self.get_mul_loss(N_mask_results, lm_labels, softmax_logits, label_masks)
        mse_loss = self.get_mse_loss(all_log_softmax_logits, lm_labels)
        # mi_loss = self.get_mi_loss(label_masks, softmax_logits, all_log_softmax_logits)

        # loss = mul_loss + mse_loss + mi_loss
        loss = mul_loss + mse_loss
        return loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        encoder_outputs = self.model.encoder(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             inputs_embeds=inputs_embeds,
                                             head_mask=head_mask,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states,
                                             return_dict=return_dict)

        encoder_last_hidden_states = encoder_outputs.last_hidden_state

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right(将labels右移一位，以生成解码器的输入)
            decoder_input_ids = self.model._shift_right(labels)

        # If decoding with past key value states, only the last tokens should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.model.decoder(input_ids=decoder_input_ids,  # label往右移得到的结果
                                             attention_mask=decoder_attention_mask,
                                             inputs_embeds=decoder_inputs_embeds,
                                             past_key_values=past_key_values,
                                             encoder_hidden_states=encoder_last_hidden_states,
                                             encoder_attention_mask=attention_mask,
                                             head_mask=decoder_head_mask,
                                             # cross_attn_head_mask=cross_attn_head_mask,
                                             use_cache=use_cache,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states,
                                             return_dict=return_dict, )

        decoder_last_layer_hiddens = decoder_outputs.last_hidden_state

        all_logits = self.vague_matrix_generation(decoder_last_layer_hiddens)  # Size([5, 8, 128, 32101])

        N_mask_results = self.vague_samples_acquisition(all_logits, labels)  # Size([5, 8, 128, 32101])

        loss = self.compute_loss(all_logits, N_mask_results, labels)

        main_pred = Seq2SeqLMOutput(loss=loss, logits=all_logits)

        at_pred = self.at_learner(encoder_last_hidden_states, attention_mask)
        ot_pred = self.ot_learner(encoder_last_hidden_states, attention_mask)
        sp_pred = self.sp_learner(encoder_last_hidden_states, attention_mask)

        masked_encoder_last_state = torch.mul(encoder_last_hidden_states, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_encoder_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        return main_pred, encoder_last_hidden_states, decoder_last_layer_hiddens, sp_pred, ot_pred, at_pred, pooled_encoder_layer

    def _step(self, batch):
        # print(f'keys : {batch.keys()}')  # keys : dict_keys(['input_ids', 'attention_mask', 'labels', 'examples', 'ac_grid_label', 'ac_grid_mask'])

        tar_labels = torch.clone(batch["labels"])
        tar_labels[tar_labels[:, :] == self.tokenizer.pad_token_id] = -100

        ac_mask_matrix = batch["ac_grid_mask"]
        ac_mask_matrix_minus_1 = [1 - x for x in batch["ac_grid_mask"]]

        if self.current_epoch < self.args.stat_full_train_ep:
            tar_labels = tar_labels * ac_mask_matrix + (ac_mask_matrix_minus_1 * -100)

        outputs, encoder_last_state, decoder_hidden_states, sp_pred, ot_pred, at_pred, pooled_encoder_layer = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=tar_labels)

        criterion = SupConLoss(loss_scaling_factor=self.args.cont_loss,
                               temperature=self.args.cont_temp)  # define loss with a temperature `temp`

        contrastive_labels = batch['contrastive_labels']
        sp_labels = contrastive_labels['sp']
        at_labels = contrastive_labels['at']
        ot_labels = contrastive_labels['ot']

        grid_labels = batch['ac_grid_label']

        loss = outputs[0]

        if self.ac_loss_lambda > 0:
            lm_logits = outputs[1]
            avg_lm_logits = lm_logits.mean(dim=0)
            ac_loss = CrossEntropyLoss(ignore_index=-100)
            loss += (ac_loss(avg_lm_logits.view(-1, avg_lm_logits.size(-1)),
                             tar_labels.view(-1)) * self.args.ac_loss_lambda)

        if self.sp_loss_lambda > 0:
            sp_summed = sp_pred
            sp_normed = normalize(sp_summed, p=2.0, dim=2)
            sp_contrastive_loss = criterion(sp_normed, sp_labels) * self.sp_loss_lambda

            at_summed = at_pred
            at_normed = normalize(at_summed, p=2.0, dim=2)
            at_contrastive_loss = criterion(at_normed, at_labels) * self.sp_loss_lambda

            ot_summed = ot_pred
            ot_normed = normalize(ot_summed, p=2.0, dim=2)
            ot_contrastive_loss = criterion(ot_normed, ot_labels) * self.sp_loss_lambda

            loss += ot_contrastive_loss + sp_contrastive_loss + at_contrastive_loss

        if self.grid_loss_lambda > 0:
            loss += (self.grid_learner(encoder_last_state,
                                       decoder_hidden_states,
                                       batch["attention_mask"],
                                       batch["ac_grid_mask"],
                                       grid_labels) *
                     self.grid_loss_lambda)

        return loss, outputs

    def training_step(self, batch, batch_idx):
        # print('training step batch :', batch.keys())  # training step batch : dict_keys(['input_ids', 'attention_mask', 'labels', 'examples'])
        # print(torch.cuda.memory_summary())

        loss, _ = self._step(batch)

        # print(torch.cuda.memory_summary())

        self.log("train_loss", loss)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx, num_beams=4)
        self.test_step_outputs.append(output)

    def eval_step(self, batch, batch_idx, num_beams=1):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
        generateds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        candidates = generateds
        if num_beams > 1:
            candidates = [generateds[i:i + num_beams] for i in range(0, len(generateds), num_beams)]
            generateds = [generateds[i] for i in range(0, len(generateds), num_beams)]

        return {
            'examples': batch['examples'],
            'predictions': generateds,
            'candidates': candidates
        }

    def validation_step(self, batch, batch_idx):
        output = self.eval_step(batch, batch_idx)  # 执行验证步骤
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        self.current_val_result = Result.parse_from(self.validation_step_outputs)  # 解析验证结果
        self.current_val_result.cal_metric()  # 计算指标

        self.update_result = False
        if (not hasattr(self, 'best_val_result')) or (self.best_val_result < self.current_val_result):
            self.best_val_result = self.current_val_result
            self.update_result = True

            # select model by devlopment set
            self.save_model(args)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.test_result = Result.parse_from(self.test_step_outputs)
        self.test_result.cal_metric()
        self.test_result.save_metric(
            self.args.output_dir,
            self.args.model_name_or_path,
            self.args.subname,
            self.args.dataset,
            self.args.seed,
            self.args.learning_rate,
        )
        self.test_result.save_prediction(
            self.args.output_dir,
            self.args.model_name_or_path,
            self.args.subname,
            self.args.dataset,
            self.args.seed,
            self.args.learning_rate,
        )
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()  # 清除缓存
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            generated_outputs = self.model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=100,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            min_con, avg_con = self.get_confidence(generated_outputs)

            index = (min_con > self.args.min_con_thre) * (avg_con > self.args.avg_con_thre)

            input_ids = batch['input_ids'][index]
            attention_mask = batch['attention_mask'][index]

            examples = batch['examples']
            examples = [examples[i] for i in range(len(examples)) if index[i]]

            min_con = min_con[index]
            avg_con = avg_con[index]

            num_beams = 4
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=num_beams,
                num_beams=num_beams,
            )

            generateds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_beams = [generateds[i:i + num_beams] for i in range(0, len(generateds), num_beams)]

        return {
            'examples': examples,
            'predictions': generated_beams,
            'min_con': min_con,
            'avg_con': avg_con,
        }

    def get_confidence(self, generated_outputs):
        input_ids = generated_outputs['sequences']
        attention_mask = self.get_mask(input_ids)[:, 1:]  # 1: to remove decoder_start_id

        probs = torch.stack(generated_outputs.scores, dim=1)
        probs = F.log_softmax(probs, dim=-1)
        confidence = probs.max(dim=-1)[0]

        confidence[~attention_mask.bool()] = 0
        min_confidence = confidence.min(dim=-1)[0].exp().detach().cpu().numpy()

        avg_confidence = confidence.sum(dim=-1) / attention_mask.sum(dim=-1)
        avg_confidence = avg_confidence.exp().detach().cpu().numpy()

        return min_confidence, avg_confidence

    def get_mask(self, input_ids):
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id

        eos_flag = (input_ids == eos_token_id)
        eos_flag = torch.cat([eos_flag[:, :1], eos_flag[:, :-1]], dim=1)
        attention_mask = torch.cumsum(eos_flag, dim=1)
        attention_mask = (attention_mask == 0).bool()

        return attention_mask.long()

    def configure_optimizers(self):

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_epsilon,
            no_deprecation_warning=True
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]


class CustomWriter(BasePredictionWriter):
    def __init__(self, argument_parser, name_space, write_interval='epoch'):
        super().__init__(write_interval)
        self.argument_parser = argument_parser
        self.name_space = name_space
        self.output_dir = name_space.model.output_dir

    def on_validation_end(self, trainer, pl_module):
        if not pl_module.update_result:
            return

        if hasattr(pl_module, 'current_train_result'):
            pl_module.current_train_result.report()
        print('------------------------------------------------------------')
        print('[current]', end=' ')
        pl_module.current_val_result.report()

        print('[best]   ', end=' ')
        pl_module.best_val_result.report()
        print('------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        output_examples = []
        N = 0
        for output in tqdm(predictions):
            examples = output['examples']
            predictions = output['predictions']
            min_confidence = output['min_con']
            avg_confidence = output['avg_con']

            for example, prediction, min_con, avg_con in zip(examples, predictions, min_confidence, avg_confidence):
                output_examples.append({
                    'ID': example['ID'],
                    'sentence': example['sentence'],
                    'quad_preds': prediction,
                    'quads_seq': example.get('quads_seq'),
                    'min_con': float(min_con),
                    'avg_con': float(avg_con),
                    'full_review': example['full_review'],
                })

        print(f'save {len(output_examples)} to', self.output_dir)
        if len(output_examples) > 10_000:
            save_line_json(output_examples, self.output_dir)
        else:
            save_json(output_examples, self.output_dir)



def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_mode', type=str, default='pseudo_labeling',
                        choices=['train_quad', 'pseudo_labeling', 'train_scorer', 'do_filtering',
                                 'train_quad_with_data_augmentation', 'do_reranking'])

    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)

    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--enable_checkpointing", type=bool, default=False)
    parser.add_argument("--enable_model_summary", type=bool, default=False)

    parser.add_argument("--max_epochs", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=-1)
    parser.add_argument("--model_name_or_path", type=str,
                        default='')  # "/ssd-data1/cxf2022/00.model_base/t5_base" if train else "../output/quad/model/acos/dataset=rest16,b='quad',seed=42"
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="acos/rest16",
                        choices=['acos/rest16', 'acos/laptop16', 'asqp/rest16', 'asqp/rest15'])
    parser.add_argument("--data_dir", type=str, default='')  # "data/t5" if train else "data/raw/yelp/100k_1.json"
    parser.add_argument('--output_dir', type=str, default='../output/quad/')
    parser.add_argument("--subname", type=str, default='', choices=['quad', '10-40_10000'])

    parser.add_argument("--stat_full_train_ep", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_clip_val", type=int, default=0)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default='', choices=["train_test", "train", "predict"])
    parser.add_argument("--self_training_data_dir", type=str, default='',
                        choices=['', '../output/filter/acos/rest16.json'])
    parser.add_argument("--filter_setting", type=str, default='node', choices=['none', '10-40_10000'])

    # sl model param
    parser.add_argument('--mc_forward_num', type=int, default=5)
    parser.add_argument("--min_con_thre", type=int, default=0.7)
    parser.add_argument("--avg_con_thre", type=int, default=0.9)
    parser.add_argument("--ac_loss_lambda", type=int, default=0.2)
    parser.add_argument("--sp_loss_lambda", type=int, default=0.2)
    parser.add_argument("--grid_loss_lambda", type=int, default=0.4)
    parser.add_argument("--cont_loss", type=int, default=0.4)
    parser.add_argument("--cont_temp", type=int, default=0.25)
    parser.add_argument("--gama", type=int, default=10)
    parser.add_argument("--m", type=int, default=-0.6)

    parser.add_argument('--use_sl', type=bool, default=True)

    args = parser.parse_args()

    if args.train_mode == 'train_quad':
        args.do_train = True
        args.do_test = True
        args.mode = "train_test"
        args.subname = 'quad'
        args.data_dir = "data/t5/"
        args.output_dir = "../output/quad/"
        args.max_epochs = 20
        args.gradient_clip_val = 1
        args.check_val_every_n_epoch = 1
        args.model_name_or_path = "/ssd-data1/cxf2022/00.model_base/t5_large"
        args.train_batch_size = 8
        args.eval_batch_size = 32
        args.weight_decay = 0.01

    elif args.train_mode == 'pseudo_labeling':
        args.do_predict = True
        args.mode = 'predict'
        args.subname = 'quad'
        if args.dataset in ("asqp/rest16", "asqp/rest15", "acos/rest16"):
            args.date_dir = "data/raw/yelp/100k_1.json"
        elif args.dataset == "acos/laptop16":
            args.date_dir = "data/raw/laptop/100k_1.json"
        args.max_seq_length = 100
        args.eval_batch_size = 10
        args.model_name_or_path = f"../output/quad/model/dataset={args.dataset},b={args.subname},seed={args.seed}"
        args.output_dir = f"../output/quad/pseudo_labeled/{args.dataset}.json"

    elif args.train_mode == 'train_scorer':
        args.do_train = True
        args.max_seq_length = 100
        args.weight_decay = 0.01
        args.eval_batch_size = 64
        args.max_epochs = 10
        args.use_ai_preference = True
        args.data_dir = "data/t5/"
        args.preference_data_dir = "data/comp/"
        args.output_dir = "../output/scorer/"
        args.subname = '10-40_10000'
        args.filter_setting = '10-40_10000'
        args.self_training_data_dir = f'../output/filter/{args.dataset}.json'

    elif args.train_mode == 'do_filtering':
        args.do_train = True
        args.max_seq_length = 100
        args.eval_batch_size = 80
        args.subname = "scorer"
        args.data_dir = "../output/quad/pseudo_labeled/${dataset}.json"
        args.model_name_or_path = "../output/scorer/model/dataset=${dataset},b=${subname},seed=42"
        args.output_dir = "../output/filter/"

    elif args.train_mode == 'train_quad_with_data_augmentation':
        args.do_train = True
        args.do_test = True
        args.subname = '10-40_10000'
        args.filter_setting = '10-40_10000'
        args.self_training_data_dir = '../output/filter/acos/rest16.json'

    elif args.train_mode == 'do_reranking':
        args.do_predict = True
        args.data_dir = "../output/quad/${date}/${dataset}_${quad_subname}_${seed}.json"
        args.model_name_or_path = f"../output/scorer/model/dataset={args.dataset},b={args.subname},seed=42"
        args.output_dir = "../output/rerank/"
        args.max_seq_length = 100
        args.eval_batch_size = 20
    else:
        print(f'wrong train_mode!')

    return args


if __name__ == '__main__':
    args = init_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.devices}'
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, legacy=True)
    dataloader = DataModule(args)

    Model = CADModule(args)

    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=1)],  # 使用单个进度条回调
        enable_progress_bar=True,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        fast_dev_run=False,
        precision='bf16-mixed')

    if args.do_train:
        print("\n-------------Conducting Training-------------")
        trainer.fit(Model, dataloader)
    if args.do_test:
        print("\n---------- Conduct test on trained checkpoint ----------")
        trainer.test(Model, dataloader)
    if args.do_predict:
        print("\n---------- Conduct predict on trained checkpoint ----------")
        trainer.predict(Model, dataloader)