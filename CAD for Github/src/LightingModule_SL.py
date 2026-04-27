import math
import warnings
import os
import time
import copy
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqLMOutput)

from utils import params_count
# from utils.data_utils import get_dataset
from utils.loss import SupConLoss

from utils import auto_init
_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class element_learner(nn.Module):
    """ Linear models used for the aspect/opinion/sentiment-specific representations """

    def __init__(self, model_name_or_path=None):
        super().__init__()
        self.in_features = 768 if model_name_or_path == 't5_base' else 1024
        self.layer_1 = nn.Linear(self.in_features, 1024)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
            Returns an encoding of input X and a simple dropout-perturbed version of X
            For use in the SupConLoss calculation
            attention_mask.size() : torch.Size([6, 128])
            x : torch.Size([6, 128, 768])
        """
        # print(f'in_features: {self.in_features}')

        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_state = torch.sum(last_state, dim=1)
        features_dropped = self.dropout(features_state)
        return torch.stack((self.layer_1(features_state), self.layer_1(features_dropped)), 1)




class grid_learner(nn.Module):
    """ Linear models used for the aspect/opinion/sentiment-specific representations """

    def __init__(self, model_name_or_path=None):
        super().__init__()
        self.in_features = 768 if model_name_or_path == 't5_base' else 1024
        self.layer_e = nn.Linear(self.in_features, 1024)
        self.layer_d = nn.Linear(self.in_features, 1024)
        self.layer_1 = nn.Linear(2048, 3)

        self.dropout = nn.Dropout(0.1)

    def forward(self, encoder_last_hidden_state, decoder_hidden_states, encoder_mask, decoder_mask, grid_labels):
        '''
        _ 表示不固定
        encoder_last_hidden_state: torch.Size([16, _, 768])
        decoder_hidden_states: torch.Size([16, _, 768])
        encoder_mask: torch.Size([16, 24])
        decoder_mask: torch.Size([16, 47])
        grid_labels: torch.Size([16, 58, 47])
        '''
        # print('encoder_last_hidden_state.size() :', encoder_last_hidden_state.size())
        # print('decoder_hidden_states.size() :', decoder_hidden_states.size())
        # print('encoder_mask.size() :', encoder_mask.size())
        # print('decoder_mask.size() :', decoder_mask.size())
        # print('grid_labels.size() :', grid_labels.size())

        e_s_len = encoder_last_hidden_state.shape[1]
        d_s_len = decoder_hidden_states.shape[1]

        decoder_hidden_state = decoder_hidden_states[-1]

        encoder_state = torch.nn.functional.relu(self.layer_e(encoder_last_hidden_state))
        decoder_state = torch.nn.functional.relu(self.layer_d(decoder_hidden_state))

        decode_last_state = torch.mul(decoder_state, decoder_mask.unsqueeze(-1))
        encode_last_state = torch.mul(encoder_state, encoder_mask.unsqueeze(-1))

        encode_last_state = encode_last_state.unsqueeze(1).expand(-1, d_s_len, -1, -1)
        decode_last_state = decode_last_state.unsqueeze(2).expand(-1, -1, e_s_len, -1)

        size_1_2_of_grid_labels = int(grid_labels.size(1)) * int(grid_labels.size(2))

        grid_state = torch.cat([decode_last_state, encode_last_state], dim=3)
        grid_state = self.layer_1(grid_state)

        grid_state = grid_state.view(16, -1)

        adaptive_adjuster = nn.Linear(grid_state.size(1), size_1_2_of_grid_labels).to('cuda:0')

        features_state = adaptive_adjuster(grid_state)
        features_dropped = self.dropout(features_state)
        features_dropped = features_dropped.reshape(16, grid_labels.size(1), grid_labels.size(2)).float()
        grid_labels = grid_labels.float().to('cuda:0')
        assert features_dropped.size() == grid_labels.size()

        grid_loss_fct = CrossEntropyLoss(ignore_index=-100)

        # features_dropped = features_dropped.view(16, -1, features_dropped.size(-1))
        # grid_labels = grid_labels.view(16, -1, features_dropped.size(-1))

        grid_loss = grid_loss_fct(features_dropped, grid_labels)

        return grid_loss



class LightningModule_SL(pl.LightningModule):

    def __init__(self, args):
        super(self).__init__()
        # self.args = args
        self.ac_loss_lambda = 0.2
        self.sp_loss_lambda = 0.2
        self.grid_loss_lambda = 0.4


        self.grid_learner = grid_learner
        self.at_learner = element_learner
        self.ac_learner = element_learner
        self.ot_learner = element_learner
        self.sp_learner = element_learner

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    def save_model(self, args):
        dir_name = os.path.join(args.output_dir, 'model', f'dataset={args.dataset},b={args.subname},seed={args.seed}')
        print(f'## save model to {dir_name}')

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
                mask_results[i, j, :all_lengths[j]] = mask_results[i, j, :all_lengths[j]].scatter(-1, all_c[i, j, :all_lengths[j]].unsqueeze(-1), 1)

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
                                                return_dict=return_dict,)

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
        print(batch)

        tar_labels = torch.clone(batch["target_ids"])
        tar_labels[tar_labels[:, :] == self.tokenizer.pad_token_id] = -100

        if self.current_epoch < self.args.stat_full_train_ep:
            ac_mask_matrix = batch["category_mask_matrix"]
            ac_mask_matrix = 1 - ac_mask_matrix
            tar_labels = tar_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)

        outputs, encoder_last_state, decoder_hidden_states, sp_pred, ot_pred, at_pred, pooled_encoder_layer = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=tar_labels,
            decoder_attention_mask=batch['target_mask'])

        # define loss with a temperature `temp`
        criterion = SupConLoss(loss_scaling_factor=self.args.cont_loss, temperature=self.args.cont_temp)

        sp_labels = batch['sentiment_labels']
        at_labels = batch['aspect_labels']
        ot_labels = batch['opinion_labels']
        grid_labels = batch['grid_matrix_label']

        loss = outputs[0]

        if self.ac_loss_lambda > 0:
            ac_mask_matrix = batch["category_mask_matrix"]
            category_labels = tar_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)
            lm_logits = outputs[1]
            ac_loss = CrossEntropyLoss(ignore_index=-100)
            loss += (ac_loss(lm_logits.view(-1, lm_logits.size(-1)), category_labels.view(-1)) *
                     self.args.ac_loss_lambda)

        if self.sp_loss_lambda > 0:
            # Calculate the characteristic-specific losses
            sp_summed = sp_pred
            sp_normed = normalize(sp_summed, p=2.0, dim=2)
            sp_contrastive_loss = criterion(sp_normed, sp_labels) * self.sp_loss_lambda
            # #print('contr_loss:\t', sp_contrastive_loss)

            at_summed = at_pred
            at_normed = normalize(at_summed, p=2.0, dim=2)
            at_contrastive_loss = criterion(at_normed, at_labels) * self.sp_loss_lambda
            # #print('as_loss:\t', at_contrastive_loss)

            ot_summed = ot_pred
            ot_normed = normalize(ot_summed, p=2.0, dim=2)
            ot_contrastive_loss = criterion(ot_normed, ot_labels) * self.sp_loss_lambda
            # print('op_loss:\t', ot_contrastive_loss)

            loss += ot_contrastive_loss + sp_contrastive_loss + at_contrastive_loss

        # return original loss plus the characteristic-specific SCL losses
        if self.grid_loss_lambda > 0:
            loss += (self.grid_learner(encoder_last_state,
                                       decoder_hidden_states,
                                       batch["source_mask"],
                                       batch["target_mask"],
                                       grid_labels) *
                     self.grid_loss_lambda)

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

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
            candidates = [generateds[i:i+num_beams] for i in range(0, len(generateds), num_beams)]
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
            self.output_dir,
            self.model_name_or_path,
            self.subname,
            self.dataset,
            self.seed,
            self.learning_rate,
        )
        self.test_result.save_prediction(
            self.output_dir,
            self.model_name_or_path,
            self.subname,
            self.dataset,
            self.seed,
            self.learning_rate,
        )
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

            index = (min_con>self.min_con_thre) * (avg_con>self.avg_con_thre)

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
            generated_beams = [generateds[i:i+num_beams] for i in range(0, len(generateds), num_beams)]

        return {
            'examples': examples,
            'predictions': generated_beams,
            'min_con': min_con,
            'avg_con': avg_con,
        }

    def get_confidence(self, generated_outputs):
        input_ids = generated_outputs['sequences']
        attention_mask = self.get_mask(input_ids)[:, 1:] # 1: to remove decoder_start_id

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
        attention_mask = torch.cumsum  (eos_flag, dim=1)
        attention_mask = (attention_mask == 0).bool()

        return attention_mask.long()

    def configure_optimizers(self, args):

        optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]