"""
CAD Framework Training Script
Implements the complete two-stage self-augmentation training pipeline.
Supports multi-dataset training for ACOS-Laptop, ACOS-Rest, ASQP-Rest15, ASQP-Rest16.
"""

import os
import sys
import time
import random
import argparse
import math

# Set CUDA_VISIBLE_DEVICES before importing torch
if '--devices' in sys.argv:
    idx = sys.argv.index('--devices')
    if idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx + 1]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import TQDMProgressBar, BasePredictionWriter

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer, T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import Seq2SeqLMOutput

# Import custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.cg_module import ControllableGeneration, LatentGridConstructor
from modules.made_module import MADEEvaluator, SelfAugmentationPipeline
from modules.diffusion import GaussianDiffusion, DiffusionForText
from utils import params_count, load_json, tokenize, load_line_json, save_line_json, save_json, auto_init
from utils.quad import make_quads_seq, parse_quads_seq, get_quad_aspect_opinion_num
from utils.quad_result import Result
from utils.loss import SupConLoss

pl.seed_everything(42)
torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Data Module
# ============================================================================

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir if args.dataset == '' else os.path.join(args.data_dir, args.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, legacy=True)
        self.raw_datasets = {}

    def load_labeled_dataset(self):
        train_file = os.path.join(self.data_dir, 'train.json')
        dev_file = os.path.join(self.data_dir, 'dev.json')
        test_file = os.path.join(self.data_dir, 'test.json')

        self.raw_datasets = {
            'train': load_json(train_file),
            'dev': load_json(dev_file),
            'test': load_json(test_file)
        }

        if self.args.self_training_data_dir and self.args.filter_setting != 'none':
            self.add_self_training_data()

        print('-----------data statistic-------------')
        for mode in ('train', 'dev', 'test'):
            num_sentences = len(self.raw_datasets[mode])
            num_quads = sum([get_quad_aspect_opinion_num(ex)[0] for ex in self.raw_datasets[mode]])
            print(f'{mode.upper():<5} | Sentences: {num_sentences:<5} | Quad: {num_quads:<5}')
        print('--------------------------------------')

    def add_self_training_data(self):
        setting, k = self.args.filter_setting.split('_')
        k = int(k)

        try:
            self_training_data = load_json(self.args.self_training_data_dir)
        except:
            self_training_data = list(load_line_json(self.args.self_training_data_dir))

        if len(self_training_data) >= 110_000:
            self_training_data = self_training_data[10_000:110_000]

        self_training_data = [{
            'ID': ex['ID'],
            'sentence': ex['sentence'],
            'quads_seq': ex['quad_preds'][0],
            'reward': ex.get('reward', [None]),
            'quad_preds': ex['quad_preds'],
        } for ex in self_training_data]

        if setting != 'full':
            try:
                start, end = setting.split('-')
                start = int(start) / 100 * len(self_training_data)
                end = int(end) / 100 * len(self_training_data)
                self_training_data = sorted(self_training_data, key=lambda e: e['reward'][0], reverse=True)[int(start):int(end)]
            except:
                raise NotImplementedError(f'Unknown setting: {self.args.filter_setting}')

        if k > 0:
            random.seed(self.args.seed)
            self_training_data = random.sample(self_training_data, k=k)
            mean_reward = sum(ex['reward'][0] for ex in self_training_data) / len(self_training_data)
            print(f'mean_reward: {mean_reward}')
            self.raw_datasets['train'].extend(self_training_data)

    def prepare_data(self):
        if self.args.mode == 'train_test':
            self.load_labeled_dataset()

    def get_dataloader(self, mode, batch_size, shuffle):
        return DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=2,
            num_workers=1,
            persistent_workers=True,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer,
                max_seq_length=self.args.max_seq_length,
                mode=mode,
                dataset=self.args.dataset
            ),
            drop_last=True if mode == 'train' else False,
        )

    def train_dataloader(self):
        return self.get_dataloader('train', self.args.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader('dev', self.args.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader('test', self.args.eval_batch_size, shuffle=False)


class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode, dataset):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.dataset = dataset

    def tok(self, text, max_seq_length):
        return tokenize(self.tokenizer, text, max_seq_length)

    def __call__(self, examples):
        examples = [ex for ex in examples if len(ex['sentence']) > 0]
        if not examples:
            return {
                'input_ids': torch.empty(0, dtype=torch.long),
                'attention_mask': torch.empty(0, dtype=torch.long),
                'labels': torch.empty(0, dtype=torch.long),
                'examples': []
            }

        text = [ex['sentence'] for ex in examples]
        batch_encodings = self.tok(text, -1)

        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        labels = None
        if self.mode in ('train', 'dev', 'test'):
            target_seqs = [self.make_quads_seq(ex) for ex in examples]
            target_encodings = self.tok(target_seqs, -1)
            labels = target_encodings['input_ids']
            labels = torch.tensor([
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels
            ])

        if self.max_seq_length > 0:
            input_ids = input_ids[:, :self.max_seq_length]
            attention_mask = attention_mask[:, :self.max_seq_length]
            if labels is not None:
                labels = labels[:, :self.max_seq_length]

        # Get contrastive labels
        contrastive_labels = {}
        if self.mode != 'predict':
            all_quads = [ex.get('quads', []) for ex in examples]
            contrastive_labels['at'] = self.get_at_labels(all_quads)
            contrastive_labels['ot'] = self.get_ot_labels(all_quads)
            contrastive_labels['sp'] = self.get_sp_labels(all_quads)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'examples': examples,
            'contrastive_labels': contrastive_labels,
        }

    def make_quads_seq(self, example):
        if 'quads_seq' in example:
            return example['quads_seq']
        return make_quads_seq(example)

    def get_at_labels(self, labels):
        at_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        at_labels = []
        for label in labels:
            at = set([quad[0] for quad in label])
            if 'NULL' not in at:
                at = at_dict['EXPLICIT']
            else:
                at = at_dict['NULL'] if len(at) == 1 else at_dict['BOTH']
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
                ot = ot_dict['NULL'] if len(ot) == 1 else ot_dict['BOTH']
            ot_labels.append(ot)
        return ot_labels

    def get_sp_labels(self, labels):
        sp_dict = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
        sp_labels = []
        for label in labels:
            sp = list(set([quad[3] for quad in label]))
            sp = sp_dict[sp[0]] if len(sp) == 1 else sp_dict['mixed']
            sp_labels.append(sp)
        return sp_labels


# ============================================================================
# Element Learner
# ============================================================================

class ElementLearner(nn.Module):
    """Linear model for aspect/opinion/sentiment-specific representations."""

    def __init__(self, in_features=1024, out_features=1024):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features = torch.sum(last_state, dim=1)
        features_dropped = self.dropout(features)
        return torch.stack((self.layer(features), self.layer(features_dropped)), 1)


# ============================================================================
# CAD Lightning Module
# ============================================================================

class CADModule(pl.LightningModule):
    """
    Complete CAD Framework Lightning Module.
    Integrates T5 backbone with CG and MADE modules.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # 记录训练时间
        self.training_start_time = None
        self.total_training_time = 0
        self.inference_times = []

        # Core T5 model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

        # Get hidden dimension
        self.hidden_dim = self.model.config.d_model

        # CG Module (可选)
        if getattr(args, 'use_cg', True):
            self.cg = ControllableGeneration(
                hidden_dim=self.hidden_dim,
                grid_dim=getattr(args, 'grid_dim', 512),
                num_categories=getattr(args, 'num_categories', 13),
                temperature=args.cont_temp,
                lambda_g=args.lambda_g,
                lambda_e=args.lambda_e,
                lambda_c=args.lambda_c,
            )
        else:
            self.cg = None

        # MADE Evaluator (可选)
        if getattr(args, 'use_made', True):
            self.made = MADEEvaluator(
                hidden_dim=self.hidden_dim,
                num_categories=getattr(args, 'num_categories', 13),
                w_consistency=args.w_consistency,
                w_diversity=args.w_diversity,
                top_k=args.top_k,
            )
        else:
            self.made = None

        # Diffusion model for generation (可选)
        if getattr(args, 'use_cg', True):
            self.diffusion = DiffusionForText(
                encoder_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                timesteps=args.diffusion_timesteps,
            )
        else:
            self.diffusion = None

        # Element learners
        self.at_learner = ElementLearner(self.hidden_dim)
        self.ot_learner = ElementLearner(self.hidden_dim)
        self.sp_learner = ElementLearner(self.hidden_dim)

        # Loss weights
        self.ac_loss_lambda = args.ac_loss_lambda
        self.sp_loss_lambda = args.sp_loss_lambda
        self.cg_loss_lambda = args.cg_loss_lambda

        # Contrastive loss
        self.contrastive_loss = SupConLoss(
            loss_scaling_factor=args.cont_loss,
            temperature=args.cont_temp
        )

        # Outputs storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

        print(f'Model initialized with {params_count(self.model):,} parameters')

    def save_model(self):
        dir_name = os.path.join(
            self.args.output_dir, 'model',
            f'dataset={self.args.dataset},b={self.args.subname},seed={self.args.seed}'
        )
        print(f'Saving model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        output_hidden_states=True,
    ):
        """Forward pass through T5 with CG integration."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        """Training step with CG and MADE integration."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        contrastive_labels = batch.get('contrastive_labels', {})

        # Forward through T5
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # Base loss
        loss = outputs.loss

        # Get encoder and decoder hidden states
        encoder_hidden = outputs.encoder_last_hidden_state
        decoder_hidden = outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None

        # Apply CG module if enabled
        if getattr(self.args, 'use_cg', True) and self.cg is not None and decoder_hidden is not None:
            # Create decoder mask
            decoder_mask = (labels != -100).float()

            # CG forward pass
            cg_loss, cg_outputs = self.cg(
                encoder_hidden=encoder_hidden,
                decoder_hidden=decoder_hidden,
                encoder_mask=attention_mask,
                decoder_mask=decoder_mask,
            )

            loss = loss + self.cg_loss_lambda * cg_loss

        # Element contrastive loss
        if self.sp_loss_lambda > 0 and contrastive_labels:
            at_pred = self.at_learner(encoder_hidden, attention_mask)
            ot_pred = self.ot_learner(encoder_hidden, attention_mask)
            sp_pred = self.sp_learner(encoder_hidden, attention_mask)

            at_loss = self.contrastive_loss(
                F.normalize(at_pred, dim=-1),
                contrastive_labels.get('at', [0] * input_ids.size(0))
            )
            ot_loss = self.contrastive_loss(
                F.normalize(ot_pred, dim=-1),
                contrastive_labels.get('ot', [0] * input_ids.size(0))
            )
            sp_loss = self.contrastive_loss(
                F.normalize(sp_pred, dim=-1),
                contrastive_labels.get('sp', [0] * input_ids.size(0))
            )

            loss = loss + self.sp_loss_lambda * (at_loss + ot_loss + sp_loss)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self.eval_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def eval_step(self, batch, batch_idx, num_beams=5):
        """Evaluation step for validation and test."""
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
        )

        generateds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if num_beams > 1:
            candidates = [generateds[i:i+num_beams] for i in range(0, len(generateds), num_beams)]
            generateds = [generateds[i] for i in range(0, len(generateds), num_beams)]
        else:
            candidates = generateds

        return {
            'examples': batch['examples'],
            'predictions': generateds,
            'candidates': candidates
        }

    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        result = Result.parse_from(self.validation_step_outputs)
        result.cal_metric()

        if not hasattr(self, 'best_val_result') or self.best_val_result < result:
            self.best_val_result = result
            self.save_model()

        print(f'Val F1: {result.detailed_metrics["f1"]:.4f}, Precision: {result.detailed_metrics["precision"]:.4f}, Recall: {result.detailed_metrics["recall"]:.4f}')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self.eval_step(batch, batch_idx, num_beams=4)
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        """Compute test metrics."""
        result = Result.parse_from(self.test_step_outputs)
        result.cal_metric()

        print(f'Test F1: {result.detailed_metrics["f1"]:.4f}, Precision: {result.detailed_metrics["precision"]:.4f}, Recall: {result.detailed_metrics["recall"]:.4f}')

        # Save results
        result.save_metric(
            self.args.output_dir,
            self.args.model_name_or_path,
            self.args.subname,
            self.args.dataset,
            self.args.seed,
            self.args.learning_rate,
        )
        result.save_prediction(
            self.args.output_dir,
            self.args.model_name_or_path,
            self.args.subname,
            self.args.dataset,
            self.args.seed,
            self.args.learning_rate,
        )

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


# ============================================================================
# Argument Parser
# ============================================================================

def init_args():
    parser = argparse.ArgumentParser(description='CAD Framework for ASQP')

    # Mode settings
    parser.add_argument('--train_mode', type=str, default='train_cad',
                        choices=['train_cad', 'train_quad', 'pseudo_labeling'])
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train_test', 'predict'])

    # Model settings
    parser.add_argument('--model_name_or_path', type=str,
                        default='/data/cxf2022/dl_project/00.model_base/t5_large')
    parser.add_argument('--max_seq_length', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--devices', type=int, default=0)

    # Data settings
    parser.add_argument('--dataset', type=str, default='acos/rest16',
                        choices=['acos/rest16', 'acos/laptop16', 'asqp/rest15', 'asqp/rest16'])
    parser.add_argument('--data_dir', type=str, default='data/t5/')
    parser.add_argument('--output_dir', type=str, default='../output/cad/')
    parser.add_argument('--subname', type=str, default='cad')
    parser.add_argument('--seed', type=int, default=42)

    # Training settings
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    # Self-training settings
    parser.add_argument('--self_training_data_dir', type=str, default='')
    parser.add_argument('--filter_setting', type=str, default='none')

    # CG Module settings
    parser.add_argument('--use_cg', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_grid_constraint', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_dual_similarity', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_char_attention', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--lambda_g', type=float, default=0.25)
    parser.add_argument('--lambda_e', type=float, default=0.25)
    parser.add_argument('--lambda_c', type=float, default=0.5)
    parser.add_argument('--cg_loss_lambda', type=float, default=0.3)
    parser.add_argument('--grid_dim', type=int, default=512)
    parser.add_argument('--num_categories', type=int, default=13)

    # MADE Module settings
    parser.add_argument('--use_made', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--w_consistency', type=float, default=0.7)
    parser.add_argument('--w_diversity', type=float, default=0.3)
    parser.add_argument('--top_k', type=int, default=4)

    # Diffusion settings
    parser.add_argument('--diffusion_timesteps', type=int, default=100)

    # Loss settings
    parser.add_argument('--ac_loss_lambda', type=float, default=0.4)
    parser.add_argument('--sp_loss_lambda', type=float, default=0.4)
    parser.add_argument('--cont_loss', type=float, default=0.6)
    parser.add_argument('--cont_temp', type=float, default=0.03)

    args = parser.parse_args()

    # Set default paths based on train_mode
    if args.train_mode == 'train_cad':
        args.do_train = True
        args.do_test = True
        args.mode = 'train_test'
        args.subname = 'cad'
        args.max_epochs = 30
        args.gradient_clip_val = 1.0
        args.learning_rate = 5e-5

    return args


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    args = init_args()

    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model_name_or_path}')

    # Create data module
    data_module = DataModule(args)

    # Create model
    model = CADModule(args)

    # Create trainer
    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # Train
    if args.mode == 'train_test':
        print('\n============= Starting Training =============')
        trainer.fit(model, data_module)

        print('\n============= Starting Testing =============')
        trainer.test(model, data_module)
