from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer
import zlib


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        # Get device from input tensor
        device = x.device

       # Convert image to tokens using tokenizer
        tokens = self.tokenizer.encode_index(x)
        
        # Store the original token shape for decompression
        shape_info = np.array(tokens.shape, dtype=np.int32).tobytes()
        
        # Add batch dimension if needed
        tokens = tokens.unsqueeze(0).to(device)  # (1, H, W)
        
        # Get token probabilities from autoregressive model
        logits, _ = self.autoregressive(tokens)
        probs = torch.softmax(logits, dim=-1)
        
        # Use predictions to encode differences (residuals)
        predicted_tokens = torch.argmax(probs, dim=-1)
        residuals = (tokens - predicted_tokens).cpu().numpy()
        
        # Compress residuals using run-length encoding + zlib
        residuals_bytes = self._compress_residuals(residuals)
        
        # Combine shape info and compressed data
        return shape_info + residuals_bytes

    def _compress_residuals(self, residuals):
        """Helper method to compress residuals efficiently"""
        # Run-length encode residuals
        rle_data = []
        current_val = residuals.flatten()[0]
        count = 1
        
        for val in residuals.flatten()[1:]:
            if val == current_val and count < 255:
                count += 1
            else:
                rle_data.extend([count, int(current_val)])
                current_val = val
                count = 1
        rle_data.extend([count, int(current_val)])
        
        # Convert to bytes and compress
        return zlib.compress(np.array(rle_data, dtype=np.int16).tobytes(), level=9)


    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        # Get device from tokenizer
        device = next(self.tokenizer.parameters()).device
        
        # Extract shape information (2 integers, 4 bytes each)
        shape_size = 2 * 4  # Two dimensions, 4 bytes each
        shape_info = np.frombuffer(x[:shape_size], dtype=np.int32)
        num_tokens_h, num_tokens_w = shape_info
        
        # Initialize output tensor
        tokens = torch.zeros((1, num_tokens_h, num_tokens_w), 
                           dtype=torch.long, device=device)
        
        # Decompress residuals
        residuals = self._decompress_residuals(x[shape_size:], num_tokens_h * num_tokens_w)
        residuals = torch.from_numpy(residuals).reshape(1, num_tokens_h, num_tokens_w).to(device)
        
        # Reconstruct tokens autoregressively
        for i in range(num_tokens_h):
            for j in range(num_tokens_w):
                # Get predictions from autoregressive model
                logits, _ = self.autoregressive(tokens)
                probs = torch.softmax(logits, dim=-1)
                
                # Get predicted token
                predicted_token = torch.argmax(probs[:, i*num_tokens_w + j])
                
                # Add residual to get actual token
                tokens[0, i, j] = (predicted_token + residuals[0, i, j]).clamp(
                    min=0, max=self.tokenizer.n_tokens-1)
        
        # Convert tokens back to image
        decompressed_image = self.tokenizer.decode_index(tokens)
        
        # Remove batch dimension to match expected shape
        decompressed_image = decompressed_image.squeeze(0)
        
        return decompressed_image


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
