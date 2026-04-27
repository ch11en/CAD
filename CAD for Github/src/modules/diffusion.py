"""
Gaussian Diffusion Model Backbone
Implements the forward and reverse diffusion processes for the CAD framework.
Based on: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion process."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model for text generation.
    Implements both forward (noising) and reverse (denoising) processes.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        model_dim: int = 1024,
        noise_dim: int = 1024,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.model_dim = model_dim
        self.noise_dim = noise_dim

        # Set up beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Register buffers for diffusion parameters
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(model_dim + noise_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, model_dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, noise_dim),
        )

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Create time step embedding."""
        # Sinusoidal positional encoding
        half_dim = self.noise_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to x_0 to get x_t at timestep t.

        Args:
            x_0: Clean data [batch_size, seq_len, model_dim]
            t: Timestep [batch_size]
            noise: Optional pre-generated noise
            mask: Optional grid mask for controlled generation

        Returns:
            x_t: Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get diffusion parameters for timestep t
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        # Apply controlled noise if mask is provided (Eq. 8-9 in paper)
        if mask is not None:
            # Weighted noise injection with grid mask
            sqrt_alpha_t = sqrt_alpha_t * (1 + 0.1 * mask)  # alpha weighting from paper

        # q(x_t | x_0) = N(x_t; sqrt(alpha_t) * x_0, (1 - alpha_t) * I)
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

        return x_t

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise from noisy data at timestep t.

        Args:
            x_t: Noisy data [batch_size, seq_len, model_dim]
            t: Timestep [batch_size]
            condition: Optional conditioning information

        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Concatenate with condition if provided
        if condition is not None:
            t_emb = t_emb + condition

        # Expand time embedding for sequence
        t_emb = t_emb[:, None, :].expand(-1, x_t.size(1), -1)

        # Predict noise
        noise_pred = self.noise_predictor(torch.cat([x_t, t_emb], dim=-1))

        return noise_pred

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Denoise x_t to get x_{t-1}.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            condition: Optional conditioning
            clip_denoised: Whether to clip predicted x_0

        Returns:
            x_{t-1}: Less noisy data
        """
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, condition)

        # Get parameters
        alpha_t = self.alphas[t][:, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t][:, None, None]

        # Predict x_0 from x_t and predicted noise
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None] * x_0_pred +
            self.posterior_mean_coef2[t][:, None, None] * x_t
        )

        # Add noise for t > 0
        if t[0] > 0:
            posterior_variance = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x_t)
            x_t_minus_1 = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t_minus_1 = posterior_mean

        return x_t_minus_1

    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Full reverse diffusion process from pure noise to data.

        Args:
            shape: Shape of the output tensor
            condition: Optional conditioning
            return_intermediates: Whether to return all intermediate steps

        Returns:
            Generated samples (and optionally intermediate steps)
        """
        batch_size = shape[0]
        device = next(self.parameters()).device

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        intermediates = [x_t] if return_intermediates else None

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, condition, clip_denoised=True)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def forward(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x_0: Clean data
            t: Optional timestep (random if not provided)
            noise: Optional pre-generated noise
            mask: Optional grid mask for controlled generation
            condition: Optional conditioning

        Returns:
            loss: Diffusion loss
            noise_pred: Predicted noise
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion: add noise
        x_t = self.q_sample(x_0, t, noise, mask)

        # Predict noise
        noise_pred = self.predict_noise(x_t, t, condition)

        # Compute loss (Eq. 14 in paper with weighted mask)
        if mask is not None:
            # Weighted MSE loss with grid mask
            weight = 1 + 0.1 * mask  # alpha from paper
            loss = F.mse_loss(noise_pred * weight, noise * weight)
        else:
            loss = F.mse_loss(noise_pred, noise)

        return loss, noise_pred


class DiffusionForText(nn.Module):
    """
    Diffusion model adapted for text generation in ASQP task.
    Works with T5 encoder representations.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 1024,
        timesteps: int = 100,
        num_layers: int = 4,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        # Project encoder output to diffusion space
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)

        # Diffusion model
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            model_dim=hidden_dim,
            noise_dim=hidden_dim,
        )

        # Project back to encoder space
        self.decoder_proj = nn.Linear(hidden_dim, encoder_dim)

        # Transformer layers for refinement
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        grid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            encoder_output: T5 encoder output [batch, seq_len, encoder_dim]
            attention_mask: Attention mask
            grid_mask: Grid mask for controlled generation

        Returns:
            loss: Diffusion loss
            reconstructed: Reconstructed encoder output
        """
        # Project to diffusion space
        x_0 = self.encoder_proj(encoder_output)

        # Apply diffusion
        loss, _ = self.diffusion(x_0, mask=grid_mask)

        # Reconstruct (for generation)
        reconstructed = self.decoder_proj(x_0)

        return loss, reconstructed

    def generate(
        self,
        batch_size: int,
        seq_len: int,
        condition: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            condition: Optional conditioning
            device: Device to generate on

        Returns:
            Generated samples
        """
        shape = (batch_size, seq_len, self.hidden_dim)

        # Generate via reverse diffusion
        samples = self.diffusion.p_sample_loop(shape, condition)

        # Apply transformer refinement
        for layer in self.transformer_layers:
            samples = layer(samples)

        # Project back to encoder space
        output = self.decoder_proj(samples)

        return output
