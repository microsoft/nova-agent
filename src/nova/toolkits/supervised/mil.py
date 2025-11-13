import torch
import torch.nn as nn
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS
from trident.slide_encoder_models import ABMILSlideEncoder

from nova.utils.deterministic import _set_deterministic

VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)

# Set deterministic behavior
_set_deterministic()

__all__ = [
    "ABMILClassificationModel",
]


class ABMILClassificationModel(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) Classification Model.

    This class implements a neural network for slide-level classification using
    attention-based multiple instance learning, suitable for histopathology and other
    weakly-supervised settings where each input (e.g., a slide or bag) is represented
    as a set of features (patches). The model consists of a feature encoder using
    attention pooling and a fully connected classifier head.

    Args:
        input_feature_dim (int): Dimensionality of input patch features (default: 768).
        n_heads (int): Number of attention heads in the encoder (default: 1).
        head_dim (int): Size of each attention head (default: 512).
        dropout (float): Dropout rate applied in the encoder (default: 0.0).
        gated (bool): Whether to use gated attention in the encoder (default: True).
        hidden_dim (int): Size of the hidden layer in the classifier (default: 256).
        num_classes (int): Number of output classes (default: 2).

    Attributes:
        feature_encoder (ABMILSlideEncoder): The attention-based feature encoder.
        classifier (nn.Sequential): The fully connected classifier head.
        num_classes (int): The number of classes.

    Methods:
        summary():
            Returns a dictionary with the number of model parameters and architecture.

        forward(x, return_raw_attention=False):
            Computes logits for classification.
            If `return_raw_attention=True`, also returns the raw attention scores.

    Model Saving:
        This model does **not** perform file I/O or save weights itself.
        To save the trained model, use PyTorch's torch.save(model.state_dict(), path).
        Common practice is to save weights to a path such as:
            <experiment_root>/fold_{i}/model_last.pt
        Example:
            torch.save(model.state_dict(), "/results/my_exp/fold_1/model_last.pt")
        To reload, use model.load_state_dict(torch.load(path)).

    Example:
        >>> model = ABMILClassificationModel(input_feature_dim=1024, num_classes=3)
        >>> x = torch.randn(1, 100, 1024)  # (batch, num_patches, feature_dim)
        >>> logits = model(x)
        >>> torch.save(model.state_dict(), "model.pt")
        >>> model.load_state_dict(torch.load("model.pt"))

    Input:
        x (torch.Tensor): Patch features with shape [batch_size, num_patches, input_feature_dim].
        return_raw_attention (bool, optional): If True, also return attention scores (default: False).

    Returns:
        logits (torch.Tensor): Classification logits, shape [batch_size, num_classes].
        attn (Optional[torch.Tensor]): Attention scores, returned only if return_raw_attention=True.

    Raises:
        None within this class, but errors may occur if input tensor shape does not match expected dimensions.

    Notes:
        - This model expects the input as a torch.Tensor of patch features per slide.
        - Attention scores can be extracted for visualization or interpretability via return_raw_attention=True.
        - Model weight saving/loading is external to this class; see example above.
    """

    def __init__(
        self,
        input_feature_dim: int = 768,
        n_heads: int = 1,
        head_dim: int = 512,
        dropout: float = 0.0,
        gated: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,  # Set to 2 for binary, >2 for multi-class
    ):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim, n_heads=n_heads, head_dim=head_dim, dropout=dropout, gated=gated
        )
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )

    def summary(self) -> dict:
        num_params = int(sum(p.numel() for p in self.parameters()))
        model_str = str(self)

        return {"num_params": num_params, "model_architecture": model_str}

    # unsure what type hints to give for the different scenarios of return_raw_attention
    def forward(self, x, return_raw_attention=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = {'features': x}  # Wrap input in a dict to match ABMILSlideEncoder's expected input format
        attn = torch.Tensor()
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features)  # shape: [batch, num_classes]

        if return_raw_attention:
            return logits, attn

        return logits
