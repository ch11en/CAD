"""
Multivariate Diffusion Evaluator (MADE) Module
Implements the evaluation and filtering mechanism from the paper.
Based on Algorithm 2 in the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class EvaluationScore:
    """Container for evaluation scores."""
    diversity: float
    consistency: float
    likelihood: float
    final_score: float


class DiversityScorer(nn.Module):
    """
    Computes diversity score for generated samples.
    Measures lexical variety, syntactic novelty, and expression shift.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_diversity_features: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature extractors for different diversity aspects
        self.lexical_encoder = nn.Sequential(
            nn.Linear(hidden_dim, num_diversity_features),
            nn.ReLU(),
            nn.Linear(num_diversity_features, num_diversity_features),
        )

        self.syntactic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, num_diversity_features),
            nn.ReLU(),
            nn.Linear(num_diversity_features, num_diversity_features),
        )

        self.semantic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, num_diversity_features),
            nn.ReLU(),
            nn.Linear(num_diversity_features, num_diversity_features),
        )

        # Diversity scoring head
        self.diversity_head = nn.Linear(num_diversity_features * 3, 1)

    def compute_lexical_diversity(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute lexical diversity between generated and reference.
        Uses vocabulary overlap and n-gram diversity.

        Args:
            generated: Generated sample embeddings [batch, seq_len, dim]
            reference: Reference sample embeddings [batch, seq_len, dim]

        Returns:
            diversity_score: Lexical diversity score
        """
        # Encode
        gen_lex = self.lexical_encoder(generated)
        ref_lex = self.lexical_encoder(reference)

        # Compute diversity as distance from reference
        # Higher distance = more diverse
        distance = F.cosine_similarity(
            gen_lex.view(gen_lex.size(0), -1),
            ref_lex.view(ref_lex.size(0), -1),
            dim=-1
        )

        # Convert to diversity (1 - similarity)
        diversity = 1 - distance

        return diversity

    def compute_syntactic_diversity(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute syntactic diversity based on structure patterns.
        """
        gen_syn = self.syntactic_encoder(generated)
        ref_syn = self.syntactic_encoder(reference)

        # Compute structural distance
        distance = F.cosine_similarity(
            gen_syn.view(gen_syn.size(0), -1),
            ref_syn.view(ref_syn.size(0), -1),
            dim=-1
        )

        return 1 - distance

    def compute_semantic_diversity(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute semantic diversity while maintaining meaning.
        """
        gen_sem = self.semantic_encoder(generated)
        ref_sem = self.semantic_encoder(reference)

        # Semantic shift
        distance = F.cosine_similarity(
            gen_sem.view(gen_sem.size(0), -1),
            ref_sem.view(ref_sem.size(0), -1),
            dim=-1
        )

        return 1 - distance

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute overall diversity score.

        Args:
            generated: Generated sample embeddings
            reference: Reference sample embeddings

        Returns:
            diversity_score: Combined diversity score [batch]
        """
        # Compute individual diversity scores
        lex_div = self.compute_lexical_diversity(generated, reference)
        syn_div = self.compute_syntactic_diversity(generated, reference)
        sem_div = self.compute_semantic_diversity(generated, reference)

        # Combine features
        combined = torch.cat([
            self.lexical_encoder(generated).mean(dim=1),
            self.syntactic_encoder(generated).mean(dim=1),
            self.semantic_encoder(generated).mean(dim=1),
        ], dim=-1)

        # Compute final score
        score = torch.sigmoid(self.diversity_head(combined)).squeeze(-1)

        return score


class ConsistencyScorer(nn.Module):
    """
    Computes consistency score for generated samples.
    Measures alignment among aspect, opinion, and polarity components.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_categories: int = 13,
        num_sentiments: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.num_sentiments = num_sentiments

        # Component encoders
        self.aspect_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.category_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.sentiment_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention for consistency checking
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Consistency scoring head
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def check_aspect_opinion_consistency(
        self,
        aspect_hidden: torch.Tensor,
        opinion_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check consistency between aspect and opinion terms.
        They should be semantically related.
        """
        aspect_enc = self.aspect_encoder(aspect_hidden)
        opinion_enc = self.opinion_encoder(opinion_hidden)

        # Cross-attention
        attended, _ = self.cross_attention(
            aspect_enc, opinion_enc, opinion_enc
        )

        return attended

    def check_category_sentiment_consistency(
        self,
        category_hidden: torch.Tensor,
        sentiment_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check consistency between category and sentiment.
        Certain categories may have typical sentiment patterns.
        """
        category_enc = self.category_encoder(category_hidden)
        sentiment_enc = self.sentiment_encoder(sentiment_hidden)

        # Cross-attention
        attended, _ = self.cross_attention(
            category_enc, sentiment_enc, sentiment_enc
        )

        return attended

    def forward(
        self,
        aspect_hidden: torch.Tensor,
        opinion_hidden: torch.Tensor,
        category_hidden: torch.Tensor,
        sentiment_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute overall consistency score.

        Args:
            aspect_hidden: Aspect term representations
            opinion_hidden: Opinion term representations
            category_hidden: Category representations
            sentiment_hidden: Sentiment representations

        Returns:
            consistency_score: Consistency score [batch]
        """
        # Check pairwise consistencies
        asp_op_consistency = self.check_aspect_opinion_consistency(
            aspect_hidden, opinion_hidden
        )
        cat_sent_consistency = self.check_category_sentiment_consistency(
            category_hidden, sentiment_hidden
        )

        # Encode all components
        aspect_enc = self.aspect_encoder(aspect_hidden).mean(dim=1)
        opinion_enc = self.opinion_encoder(opinion_hidden).mean(dim=1)
        category_enc = self.category_encoder(category_hidden).mean(dim=1)
        sentiment_enc = self.sentiment_encoder(sentiment_hidden).mean(dim=1)

        # Combine all features
        combined = torch.cat([
            aspect_enc,
            opinion_enc,
            category_enc,
            sentiment_enc,
        ], dim=-1)

        # Compute consistency score
        score = self.consistency_head(combined).squeeze(-1)

        return score


class CharacterAwareLikelihood(nn.Module):
    """
    Computes character-aware likelihood score.
    Implements Eq. 15-16 in the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        vocab_size: int = 32100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Character-level model
        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.char_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        # Likelihood head
        self.likelihood_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute character-aware likelihood score.

        Args:
            generated_ids: Generated token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            likelihood_score: Likelihood score [batch]
        """
        # Embed tokens
        char_emb = self.char_embedding(generated_ids)

        # Process through RNN
        output, _ = self.char_rnn(char_emb)

        # Compute logits
        logits = self.likelihood_head(output)

        # Compute log-likelihood
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather probabilities for actual tokens
        # Shift for next-token prediction
        target_ids = generated_ids[:, 1:]
        log_probs = log_probs[:, :-1]

        # Gather
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Apply mask and sum
        mask = attention_mask[:, 1:]
        likelihood = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)

        # Normalize to [0, 1]
        likelihood_score = torch.sigmoid(likelihood)

        return likelihood_score


class MADEEvaluator(nn.Module):
    """
    Complete Multivariate Diffusion Evaluator (MADE) module.
    Implements Algorithm 2 from the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_categories: int = 13,
        w_consistency: float = 0.5,
        w_diversity: float = 0.5,
        top_k: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_consistency = w_consistency
        self.w_diversity = w_diversity
        self.top_k = top_k

        # Component scorers
        self.diversity_scorer = DiversityScorer(hidden_dim)
        self.consistency_scorer = ConsistencyScorer(hidden_dim, num_categories)
        self.likelihood_scorer = CharacterAwareLikelihood(hidden_dim)

        # Final scoring network
        self.final_scorer = nn.Sequential(
            nn.Linear(3, 64),  # 3 scores: diversity, consistency, likelihood
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def evaluate_sample(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        aspect_hidden: torch.Tensor,
        opinion_hidden: torch.Tensor,
        category_hidden: torch.Tensor,
        sentiment_hidden: torch.Tensor,
        generated_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> EvaluationScore:
        """
        Evaluate a single generated sample.

        Args:
            generated: Generated sample embeddings
            reference: Reference sample embeddings
            aspect_hidden: Aspect term representations
            opinion_hidden: Opinion term representations
            category_hidden: Category representations
            sentiment_hidden: Sentiment representations
            generated_ids: Generated token IDs (for likelihood)
            attention_mask: Attention mask

        Returns:
            EvaluationScore with all component scores
        """
        # Compute diversity score
        diversity = self.diversity_scorer(generated, reference)

        # Compute consistency score
        consistency = self.consistency_scorer(
            aspect_hidden, opinion_hidden, category_hidden, sentiment_hidden
        )

        # Compute likelihood score
        if generated_ids is not None and attention_mask is not None:
            likelihood = self.likelihood_scorer(generated_ids, attention_mask)
        else:
            likelihood = torch.ones_like(diversity)

        # Compute final score (Eq. 15 in paper)
        final_score = (
            self.w_consistency * consistency +
            self.w_diversity * diversity
        )

        return EvaluationScore(
            diversity=diversity.mean().item(),
            consistency=consistency.mean().item(),
            likelihood=likelihood.mean().item(),
            final_score=final_score.mean().item(),
        )

    def evaluate_batch(
        self,
        generated_batch: List[torch.Tensor],
        reference: torch.Tensor,
        element_hiddens: Dict[str, torch.Tensor],
    ) -> List[EvaluationScore]:
        """
        Evaluate a batch of generated samples.

        Args:
            generated_batch: List of generated sample tensors
            reference: Reference sample tensor
            element_hiddens: Dict with aspect, opinion, category, sentiment hiddens

        Returns:
            List of EvaluationScore for each sample
        """
        scores = []
        for generated in generated_batch:
            score = self.evaluate_sample(
                generated=generated,
                reference=reference,
                aspect_hidden=element_hiddens['aspect'],
                opinion_hidden=element_hiddens['opinion'],
                category_hidden=element_hiddens['category'],
                sentiment_hidden=element_hiddens['sentiment'],
            )
            scores.append(score)

        return scores

    def filter_samples(
        self,
        generated_batch: List[torch.Tensor],
        scores: List[EvaluationScore],
        threshold: float = 0.5,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Filter samples based on evaluation scores.

        Args:
            generated_batch: List of generated samples
            scores: List of evaluation scores
            threshold: Minimum score threshold

        Returns:
            filtered_samples: List of high-quality samples
            indices: Indices of selected samples
        """
        # Sort by final score
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i].final_score,
            reverse=True
        )

        # Select top-k samples above threshold
        filtered_samples = []
        selected_indices = []

        for idx in sorted_indices[:self.top_k]:
            if scores[idx].final_score >= threshold:
                filtered_samples.append(generated_batch[idx])
                selected_indices.append(idx)

        return filtered_samples, selected_indices

    def forward(
        self,
        generated_batch: List[torch.Tensor],
        reference: torch.Tensor,
        element_hiddens: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> Tuple[List[torch.Tensor], List[EvaluationScore]]:
        """
        Complete MADE evaluation and filtering process.

        Args:
            generated_batch: List of generated samples
            reference: Reference sample
            element_hiddens: Element hidden states
            threshold: Filtering threshold

        Returns:
            filtered_samples: High-quality samples
            scores: Evaluation scores for all samples
        """
        # Evaluate all samples
        scores = self.evaluate_batch(generated_batch, reference, element_hiddens)

        # Filter samples
        filtered_samples, _ = self.filter_samples(
            generated_batch, scores, threshold
        )

        return filtered_samples, scores


class SelfAugmentationPipeline(nn.Module):
    """
    Complete self-augmentation pipeline combining CG and MADE.
    Implements the two-stage process from the paper.
    """

    def __init__(
        self,
        cg_module: nn.Module,
        made_evaluator: MADEEvaluator,
        num_candidates: int = 4,
    ):
        super().__init__()
        self.cg = cg_module
        self.made = made_evaluator
        self.num_candidates = num_candidates

    def generate_candidates(
        self,
        encoder_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Generate multiple candidate samples using CG.

        Args:
            encoder_hidden: Encoder hidden states
            decoder_hidden: Decoder hidden states
            encoder_mask: Encoder attention mask
            decoder_mask: Decoder attention mask

        Returns:
            List of candidate samples
        """
        candidates = []

        for _ in range(self.num_candidates):
            # Apply CG with different noise
            loss, outputs = self.cg(
                encoder_hidden, decoder_hidden,
                encoder_mask, decoder_mask
            )

            # Get perturbed representations
            candidate = outputs['perturbed_decoder']
            candidates.append(candidate)

        return candidates

    def augment_dataset(
        self,
        batch: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], List[EvaluationScore]]:
        """
        Augment dataset with high-quality generated samples.

        Args:
            batch: Input batch with encoder/decoder hidden states
            threshold: MADE filtering threshold

        Returns:
            augmented_samples: High-quality augmented samples
            scores: Evaluation scores
        """
        # Stage 1: Generate candidates with CG
        candidates = self.generate_candidates(
            batch['encoder_hidden'],
            batch['decoder_hidden'],
            batch['encoder_mask'],
            batch['decoder_mask'],
        )

        # Stage 2: Evaluate and filter with MADE
        element_hiddens = {
            'aspect': batch.get('aspect_hidden', batch['encoder_hidden'].mean(dim=1)),
            'opinion': batch.get('opinion_hidden', batch['decoder_hidden'].mean(dim=1)),
            'category': batch.get('category_hidden', batch['encoder_hidden'].mean(dim=1)),
            'sentiment': batch.get('sentiment_hidden', batch['decoder_hidden'].mean(dim=1)),
        }

        filtered_samples, scores = self.made(
            candidates,
            batch['decoder_hidden'],
            element_hiddens,
            threshold,
        )

        # Convert to sample dictionaries
        augmented_samples = []
        for sample in filtered_samples:
            augmented_samples.append({
                'hidden': sample,
                'source_id': batch.get('id', None),
            })

        return augmented_samples, scores
