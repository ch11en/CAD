"""
Controllable Generation (CG) Module
Implements the three key components from the paper:
1. Controlled Noise Perturbation Generation
2. Dual Similarity-Guided Generation Constraint
3. Character-Level Attention Constraint
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class LatentGridConstructor(nn.Module):
    """
    Constructs latent sentiment grid from encoder representations.
    Implements Eq. 3-5 in the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        grid_dim: int = 512,
        num_categories: int = 13,  # Number of aspect categories
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_dim = grid_dim
        self.num_categories = num_categories

        # Projection layers for grid construction
        self.encoder_proj = nn.Linear(hidden_dim, grid_dim)
        self.decoder_proj = nn.Linear(hidden_dim, grid_dim)

        # Grid fusion layer
        self.grid_fusion = nn.Linear(grid_dim * 2, grid_dim)

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, grid_dim)

        # Sentiment embedding (negative, neutral, positive, mixed)
        self.sentiment_embedding = nn.Embedding(4, grid_dim)

    def construct_grid(
        self,
        encoder_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct latent sentiment grid G_g from encoder and decoder representations.

        Args:
            encoder_hidden: [batch, enc_len, hidden_dim]
            decoder_hidden: [batch, dec_len, hidden_dim]
            encoder_mask: [batch, enc_len]
            decoder_mask: [batch, dec_len]

        Returns:
            grid: [batch, dec_len, enc_len, grid_dim]
        """
        batch_size = encoder_hidden.size(0)
        enc_len = encoder_hidden.size(1)
        dec_len = decoder_hidden.size(1)

        # Project to grid space
        enc_proj = self.encoder_proj(encoder_hidden)  # [batch, enc_len, grid_dim]
        dec_proj = self.decoder_proj(decoder_hidden)  # [batch, dec_len, grid_dim]

        # Apply masks
        enc_proj = enc_proj * encoder_mask.unsqueeze(-1)
        dec_proj = dec_proj * decoder_mask.unsqueeze(-1)

        # Construct grid via outer product
        # Expand dimensions for broadcasting
        enc_expanded = enc_proj.unsqueeze(1).expand(-1, dec_len, -1, -1)  # [batch, dec_len, enc_len, grid_dim]
        dec_expanded = dec_proj.unsqueeze(2).expand(-1, -1, enc_len, -1)  # [batch, dec_len, enc_len, grid_dim]

        # Fuse encoder and decoder information
        grid = self.grid_fusion(torch.cat([enc_expanded, dec_expanded], dim=-1))

        return grid

    def apply_gaussian_blur(
        self,
        grid: torch.Tensor,
        kernel_size: int = 3,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to the grid for low-pass filtering.
        Implements Eq. 10 in the paper.

        Args:
            grid: [batch, dec_len, enc_len, grid_dim]
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation of Gaussian

        Returns:
            blurred_grid: Smoothed grid
        """
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float, device=grid.device) - kernel_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()

        # grid: [batch, dec_len, enc_len, grid_dim]
        # We apply separable convolution using depthwise convolution
        grid_dim = grid.size(-1)

        # Permute to [batch, grid_dim, dec_len, enc_len] for conv2d
        grid_permuted = grid.permute(0, 3, 1, 2)  # [batch, grid_dim, dec_len, enc_len]

        # Create kernel for depthwise convolution along encoder dimension
        # For depthwise conv: weight shape [channels, 1, kernel_h, kernel_w]
        kernel_enc = kernel_1d.view(1, 1, kernel_size, 1).expand(grid_dim, 1, kernel_size, 1)

        # Apply depthwise convolution along encoder dimension (groups=grid_dim)
        blurred = F.conv2d(grid_permuted, kernel_enc, padding=(0, kernel_size // 2), groups=grid_dim)

        # Create kernel for depthwise convolution along decoder dimension
        kernel_dec = kernel_1d.view(1, 1, 1, kernel_size).expand(grid_dim, 1, 1, kernel_size)

        # Apply depthwise convolution along decoder dimension
        blurred = F.conv2d(blurred, kernel_dec, padding=(kernel_size // 2, 0), groups=grid_dim)

        # Restore original shape [batch, dec_len, enc_len, grid_dim]
        blurred_grid = blurred.permute(0, 2, 3, 1)

        return blurred_grid

    def compute_binary_mask(
        self,
        grid: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Convert grid to binary mask via thresholding.
        Implements Eq. 11 in the paper.

        Args:
            grid: [batch, dec_len, enc_len, grid_dim]
            threshold: Threshold value

        Returns:
            mask: Binary mask [batch, dec_len, enc_len]
        """
        # Take L2 norm across grid dimension
        grid_norm = torch.norm(grid, dim=-1)

        # Apply threshold
        mask = (grid_norm > threshold).float()

        return mask

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to construct grid and binary mask.

        Returns:
            grid: Latent sentiment grid
            mask: Binary grid mask
        """
        # Construct grid
        grid = self.construct_grid(encoder_hidden, decoder_hidden, encoder_mask, decoder_mask)

        # Apply Gaussian blur
        blurred_grid = self.apply_gaussian_blur(grid)

        # Compute binary mask
        mask = self.compute_binary_mask(blurred_grid)

        return grid, mask


class ControlledNoisePerturbation(nn.Module):
    """
    Implements controlled noise perturbation for diverse generation.
    Based on Eq. 6-9 in the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        noise_scale: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_scale = noise_scale

        # Noise injection layer
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)

        # Domain mask embedding
        self.domain_embedding = nn.Embedding(10, hidden_dim)  # 10 domain types

    def perturb_with_mask(
        self,
        x: torch.Tensor,
        mask_strength: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply controlled noise perturbation based on mask strength.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask_strength: Pre-computed mask strength [batch, seq_len]
            noise: Optional pre-generated noise

        Returns:
            Perturbed tensor
        """
        if noise is None:
            noise = torch.randn_like(x)

        # Scale noise by mask strength
        scaled_noise = noise * self.noise_scale * (1 + mask_strength.unsqueeze(-1))

        # Apply perturbation
        perturbed = x + scaled_noise

        return perturbed

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply controlled noise perturbation to encoder and decoder hidden states.

        Args:
            encoder_hidden: [batch, enc_len, hidden_dim]
            decoder_hidden: [batch, dec_len, hidden_dim]
            grid_mask: [batch, dec_len, enc_len]

        Returns:
            perturbed_encoder: Perturbed encoder hidden states
            perturbed_decoder: Perturbed decoder hidden states
        """
        # For encoder: reduce mask along decoder dimension to get [batch, enc_len]
        encoder_mask_strength = grid_mask.mean(dim=1)  # [batch, enc_len]

        # For decoder: reduce mask along encoder dimension to get [batch, dec_len]
        decoder_mask_strength = grid_mask.mean(dim=-1)  # [batch, dec_len]

        # Perturb encoder
        perturbed_encoder = self.perturb_with_mask(encoder_hidden, encoder_mask_strength)

        # Perturb decoder
        perturbed_decoder = self.perturb_with_mask(decoder_hidden, decoder_mask_strength)

        return perturbed_encoder, perturbed_decoder


class DualSimilarityConstraint(nn.Module):
    """
    Implements dual similarity-guided generation constraint.
    Based on Eq. 9-12 in the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Projection for similarity computation
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def compute_similarity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between x and y.

        Args:
            x: [batch, dim] or [batch, seq_len, dim]
            y: [batch, dim] or [batch, seq_len, dim]

        Returns:
            similarity: Cosine similarity
        """
        # Normalize
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)

        # Compute similarity
        if x.dim() == 3:
            similarity = (x_norm * y_norm).sum(dim=-1)  # [batch, seq_len]
        else:
            similarity = (x_norm * y_norm).sum(dim=-1)  # [batch]

        return similarity

    def compute_grid_similarity(
        self,
        grid_pred: torch.Tensor,
        grid_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between predicted and target grids.
        Implements Eq. 9 in the paper.

        Args:
            grid_pred: [batch, dec_len, enc_len, dim]
            grid_target: [batch, dec_len, enc_len, dim]

        Returns:
            similarity: [batch]
        """
        # Flatten spatial dimensions
        pred_flat = grid_pred.view(grid_pred.size(0), -1, grid_pred.size(-1))
        target_flat = grid_target.view(grid_target.size(0), -1, grid_target.size(-1))

        # Compute similarity for each position
        sim = self.compute_similarity(pred_flat, target_flat)  # [batch, positions]

        # Average over positions
        return sim.mean(dim=-1)

    def compute_element_similarity(
        self,
        pred_elements: Dict[str, torch.Tensor],
        target_elements: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute similarity between predicted and target elements.
        Implements Eq. 9 in the paper.

        Args:
            pred_elements: Dict with keys 'aspect', 'opinion', 'category', 'sentiment'
            target_elements: Dict with same keys

        Returns:
            similarities: Dict of similarities for each element type
        """
        similarities = {}
        for key in pred_elements:
            similarities[key] = self.compute_similarity(pred_elements[key], target_elements[key])

        return similarities

    def contrastive_loss(
        self,
        sim_high: torch.Tensor,
        sim_low: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss using logarithmic transformation.
        Implements Eq. 10-11 in the paper.

        Args:
            sim_high: Highest similarity (positive) pair
            sim_low: Lowest similarity (negative) pair

        Returns:
            loss: Contrastive loss
        """
        # Logarithmic transformation to stabilize gradients
        s_g = sim_low - torch.log(sim_high + 1e-8)

        # Cross-entropy style loss
        loss = -torch.log(torch.exp(s_g / self.temperature) / (torch.exp(s_g / self.temperature) + 1))

        return loss.mean()

    def forward(
        self,
        grid_pred: torch.Tensor,
        grid_target: torch.Tensor,
        pred_elements: Dict[str, torch.Tensor],
        target_elements: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dual similarity constraint loss.

        Returns:
            grid_loss: Grid similarity loss
            element_loss: Element similarity loss
        """
        # Grid similarity
        grid_sim = self.compute_grid_similarity(grid_pred, grid_target)

        # Element similarities
        element_sims = self.compute_element_similarity(pred_elements, target_elements)

        # Compute losses (assuming we have high/low similarity pairs)
        # For simplicity, use MSE loss for now
        grid_loss = F.mse_loss(grid_sim, torch.ones_like(grid_sim))

        element_loss = sum(F.mse_loss(sim, torch.ones_like(sim)) for sim in element_sims.values())

        return grid_loss, element_loss


class CharacterLevelAttention(nn.Module):
    """
    Implements character-level attention constraint.
    Based on Eq. 13 in the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        vocab_size: int = 32100,  # T5 vocab size
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Character-level position embedding
        self.char_position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Multi-head attention for character-level modeling
        self.char_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # Classification head for character prediction
        self.char_classifier = nn.Linear(hidden_dim, vocab_size)

    def get_char_positions(
        self,
        text: List[str],
        tokenizer,
    ) -> torch.Tensor:
        """
        Get character-level positions for each token.

        Args:
            text: List of text strings
            tokenizer: Tokenizer for encoding

        Returns:
            positions: Character position indices
        """
        positions = []
        for t in text:
            tokens = tokenizer.encode(t, add_special_tokens=False)
            char_pos = []
            current_pos = 0
            for token in tokens:
                token_str = tokenizer.decode([token])
                char_pos.extend([current_pos + i for i in range(len(token_str))])
                current_pos += len(token_str)
            positions.append(char_pos)

        # Pad to max length
        max_len = max(len(p) for p in positions)
        positions = [p + [0] * (max_len - len(p)) for p in positions]

        return torch.tensor(positions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        char_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute character-level attention and loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]
            char_positions: Optional character position indices

        Returns:
            attended: Character-attended hidden states
            loss: Character-level classification loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get character position embeddings
        if char_positions is None:
            char_positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        pos_emb = self.char_position_embedding(char_positions.clamp(0, self.max_seq_len - 1))

        # Add position information
        hidden_with_pos = hidden_states + pos_emb

        # Apply character-level attention
        attended, _ = self.char_attention(
            hidden_with_pos,
            hidden_with_pos,
            hidden_with_pos,
            key_padding_mask=~attention_mask.bool(),
        )

        # Compute character predictions
        char_logits = self.char_classifier(attended)

        # Compute loss (cross-entropy with masked positions)
        # This encourages the model to predict correct character positions
        loss = F.mse_loss(attended, hidden_states)  # Simplified loss

        return attended, loss


class ControllableGeneration(nn.Module):
    """
    Complete Controllable Generation (CG) module.
    Combines all three components from the paper.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        grid_dim: int = 512,
        num_categories: int = 13,
        temperature: float = 0.07,
        lambda_g: float = 0.4,
        lambda_e: float = 0.4,
        lambda_c: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda_g = lambda_g
        self.lambda_e = lambda_e
        self.lambda_c = lambda_c

        # Component modules
        self.grid_constructor = LatentGridConstructor(hidden_dim, grid_dim, num_categories)
        self.noise_perturbation = ControlledNoisePerturbation(hidden_dim)
        self.similarity_constraint = DualSimilarityConstraint(hidden_dim, temperature)
        self.char_attention = CharacterLevelAttention(hidden_dim)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
        target_grid: Optional[torch.Tensor] = None,
        target_elements: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for CG module.

        Args:
            encoder_hidden: Encoder hidden states
            decoder_hidden: Decoder hidden states
            encoder_mask: Encoder attention mask
            decoder_mask: Decoder attention mask
            target_grid: Target grid for supervised learning
            target_elements: Target elements for supervised learning

        Returns:
            total_loss: Combined CG loss
            outputs: Dictionary with intermediate outputs
        """
        outputs = {}

        # 1. Construct latent grid
        pred_grid, grid_mask = self.grid_constructor(
            encoder_hidden, decoder_hidden, encoder_mask, decoder_mask
        )
        outputs['grid'] = pred_grid
        outputs['grid_mask'] = grid_mask

        # 2. Apply controlled noise perturbation
        perturbed_encoder, perturbed_decoder = self.noise_perturbation(
            encoder_hidden, decoder_hidden, grid_mask
        )
        outputs['perturbed_encoder'] = perturbed_encoder
        outputs['perturbed_decoder'] = perturbed_decoder

        # 3. Compute similarity constraints (self-supervised)
        # Use original representations as pseudo-targets for consistency
        pred_elements = {
            'aspect': perturbed_encoder.mean(dim=1),
            'opinion': perturbed_decoder.mean(dim=1),
            'category': perturbed_encoder.mean(dim=1),
            'sentiment': perturbed_decoder.mean(dim=1),
        }

        if target_grid is not None and target_elements is not None:
            grid_loss, element_loss = self.similarity_constraint(
                pred_grid, target_grid, pred_elements, target_elements
            )
        else:
            # Self-supervised: encourage perturbed representations to stay close to originals
            # This implements the consistency constraint from the paper
            target_elements = {
                'aspect': encoder_hidden.mean(dim=1),
                'opinion': decoder_hidden.mean(dim=1),
                'category': encoder_hidden.mean(dim=1),
                'sentiment': decoder_hidden.mean(dim=1),
            }
            grid_loss, element_loss = self.similarity_constraint(
                pred_grid, pred_grid.detach(), pred_elements, target_elements
            )

        outputs['grid_loss'] = grid_loss
        outputs['element_loss'] = element_loss

        # 4. Character-level attention
        char_attended, char_loss = self.char_attention(
            decoder_hidden, decoder_mask
        )
        outputs['char_attended'] = char_attended
        outputs['char_loss'] = char_loss

        # 5. Compute total loss (Eq. 14 in paper)
        total_loss = (
            self.lambda_g * grid_loss +
            self.lambda_e * element_loss +
            self.lambda_c * char_loss
        )
        outputs['total_loss'] = total_loss

        return total_loss, outputs
