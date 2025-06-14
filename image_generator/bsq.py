import abc

import torch

from .ae import PatchAutoEncoder
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        # A linear up-projection into embedding_dim should suffice
        self.project_in = torch.nn.Linear(embedding_dim, codebook_bits)
        self.project_out = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        projected = self.project_in(x)
        normalized = F.normalize(projected, p=2, dim=-1)
        return diff_sign(normalized)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.project_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self.codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.quantizer = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
                Convert input image to indices
                x: Input tensor of shape (B, H, W, 3)
                Returns: Index tensor
        """
        # Get encoded features from parent's encoder
        encoded = super().encode(x)
        # Convert to indices using BSQ
        return self.quantizer.encode_index(encoded)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert indices back to image
        x: Index tensor
        Returns: Output tensor of shape (B, H, W, 3)
        """
        # Convert indices back to features using BSQ
        decoded_features = self.quantizer.decode_index(x)
        # Decode features to image using parent's decoder
        return super().decode(decoded_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
                Convert input to binary codes
                x: Input tensor of shape (B, H, W, 3)
                Returns: Quantized tensor with -1/1 values
                """
        # Get encoded features from parent's encoder
        encoded = super().encode(x)
        # Convert to binary codes using BSQ
        #return self.quantizer.encode(encoded)
        return encoded


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
                Convert binary codes back to image
                x: Quantized tensor with -1/1 values
                Returns: Output tensor of shape (B, H, W, 3)
                """
        # Convert binary codes back to features using BSQ
        decoded_features = self.quantizer.decode(x)
        # Decode features to image using parent's decoder
        #return super().decode(decoded_features)
        return super().decode(x)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        
        """
        encoded = super().encode(x)
        # Apply BSQ quantization
        quantized = self.quantizer(encoded)
        # Decode using parent's decoder
        decoded = super().decode(quantized)
        return decoded, {} """

        encoded = super().encode(x)
        quantized_encode = self.quantizer.encode(encoded)
        quantized_decode=self.quantizer.decode(quantized_encode)
        decoded = super().decode(quantized_decode)
        return decoded, {}
