import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        # Token embedding layer
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Transformer encoder layer (used as decoder)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,  # Number of attention heads
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection to token probabilities
        self.output_projection = torch.nn.Linear(d_latent, n_tokens)
        
        # Save parameters
        self.d_latent = d_latent
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, H, W = x.shape
        
        # Flatten the spatial dimensions
        x_flat = x.reshape(B, H * W)
        
        # Embed the tokens
        embedded = self.embedding(x_flat)  # (B, H*W, d_latent)
        
        # Create causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(H * W).to(x.device)
        
        # Apply transformer with causal mask
        transformed = self.transformer(embedded, mask)
        
        # Project to token probabilities
        logits = self.output_projection(transformed)  # (B, H*W, n_tokens)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize output tensor with zeros
        output = torch.zeros((B, h * w), dtype=torch.long, device=device)
        
        # Generate tokens autoregressively
        for i in range(h * w):
            # Get current sequence
            current_sequence = output[:, :i]
            
            # Embed the sequence
            embedded = self.embedding(current_sequence)
            
            # Create causal mask for current length
            mask = torch.nn.Transformer.generate_square_subsequent_mask(i).to(device)
            
            # Get transformer output
            if i > 0:  # Only run transformer if we have inputs
                transformed = self.transformer(embedded, mask)
                # Get next token probabilities
                logits = self.output_projection(transformed[:, -1])
            else:
                # For first token, just use embedding projection
                logits = self.output_projection(torch.zeros(B, self.d_latent, device=device))
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)[:, 0]
            
            # Add to output
            output[:, i] = next_token
            
        # Reshape back to (B, h, w)
        return output.reshape(B, h, w)


        
