from __future__ import annotations

import os
import warnings
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("LM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*")


def _device() -> torch.device:
    # Try TPU first (for Colab TPU runtime)
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to GPU/MPS/CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _hidden_dim(model) -> int:
    config = getattr(model, "config", None)
    if config and hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if config and hasattr(config, "n_embd"):
        return int(config.n_embd)
    # Fallback to a small default.
    return 512


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model():
    device = _device()
    from logging import getLogger

    log = getLogger("dream")
    log.info("Loading model %s on %s", MODEL_ID, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Check if we're on TPU/XLA
    is_tpu = str(device).startswith("xla")
    
    if is_tpu:
        # TPU: Load on CPU first, then move to TPU
        # Use bfloat16 which TPU handles well
        log.info("TPU detected - loading model on CPU first, then moving to XLA device")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,  # TPU prefers bfloat16
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        log.info("Model moved to TPU: %s", device)
    else:
        # GPU/MPS/CPU: Use standard loading
        # FP32 for training stability
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto" if device.type not in ("cpu", "mps") else None,
            dtype=dtype,
        )
        if device.type == "mps":
            model = model.to(device)
    
    return tokenizer, model


class DreamCoherenceHead(nn.Module):
    def __init__(self, hidden_dim: int, num_diag: int) -> None:
        super().__init__()
        self.coherence = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.diagnostics = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_diag),
        )

    def forward(self, z_world: torch.Tensor):
        return {
            "coherence": self.coherence(z_world).squeeze(-1),
            "diagnostics": self.diagnostics(z_world),
        }


class DreamCoherenceModel(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, num_diag: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = _hidden_dim(base_model)
        self.head = DreamCoherenceHead(hidden_dim=self.hidden_dim, num_diag=num_diag)
        self.extra_layers = nn.ModuleList()
        self.aux_attention = nn.ModuleList()
        self.mlp_expansions = nn.ModuleList()
        self.lora_adapters = nn.ModuleList()
        self.new_parameters = []

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        coherence_target: torch.Tensor | None = None,
        diag_target: torch.Tensor | None = None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Hidden states required for coherence head.")
        mid_idx = len(hidden_states) // 2
        h_mid = hidden_states[mid_idx]  # [batch, seq, hidden]
        h = hidden_states[-1]
        
        # Apply adapters and expansions as residuals.
        for adapter in getattr(self, "lora_adapters", []):
            h = h + adapter(h)
        for attn in getattr(self, "aux_attention", []):
            attn_out, _ = attn(h, h, h)
            h = h + attn_out
        for mlp in getattr(self, "mlp_expansions", []):
            h = h + mlp(h)
        for layer in getattr(self, "extra_layers", []):
            h = h + layer(h)
            
        # Recalculate z_world from the *modified* h? 
        # Actually coherence head wants a global view. Let's keep using h_mid or h.
        # But for LM loss, we MUST use this modified h.
        
        losses = {}
        lm_logits = None
        
        # If we have labels, compute language modeling loss using the ADAPTED hidden state.
        # This makes the adapters trainable!
        if labels is not None:
            lm_head = self.base_model.get_output_embeddings()
            lm_logits = lm_head(h)
            
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses["lm_loss"] = lm_loss
        else:
            # If generating or just extracting features, we might still want logits
            lm_head = self.base_model.get_output_embeddings()
            lm_logits = lm_head(h)

        z_world = h.mean(dim=1) # Use the final adapted state for coherence too, better correlation.
        head_out = self.head(z_world)

        if coherence_target is not None:
            losses["coh_loss"] = nn.functional.mse_loss(
                head_out["coherence"], coherence_target.float()
            )
        if diag_target is not None:
            losses["diag_loss"] = nn.functional.binary_cross_entropy_with_logits(
                head_out["diagnostics"], diag_target.float()
            )
        if losses:
            losses["loss"] = sum(losses.values())

        return {
            "lm_logits": lm_logits if lm_logits is not None else outputs.logits,
            "hidden_states": hidden_states,
            "coherence": head_out["coherence"],
            "diagnostics": head_out["diagnostics"],
            **losses,
        }


def load_dream_model(num_diag: int | None = None):
    # Import here to avoid circular dependencies if possible, 
    # but strictly we need the length of the labels.
    from .rules import DIAG_LABELS
    if num_diag is None:
        num_diag = len(DIAG_LABELS)

    tokenizer, base_model = load_model()
    model = DreamCoherenceModel(base_model, num_diag=num_diag)
    
    # Do NOT call model.to(_device()) blindly if base_model uses device_map="auto"
    # as it might conflict with Accelerate hooks or cause meta-tensor errors.
    # Instead, we ensure our new components are on the same device AND dtype as the base model.
    try:
        base_param = next(base_model.parameters())
        dev = base_param.device
        dtype = base_param.dtype
    except StopIteration:
        dev = _device()
        dtype = torch.float32
        
    model.head.to(device=dev, dtype=dtype)
    # Also ensure any other components are moved if they were added (initially empty though)
    model.extra_layers.to(device=dev, dtype=dtype)
    model.aux_attention.to(device=dev, dtype=dtype)
    model.mlp_expansions.to(device=dev, dtype=dtype)
    model.lora_adapters.to(device=dev, dtype=dtype)
    
    return tokenizer, model
