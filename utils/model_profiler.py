"""
Model profiling utility for RUL teacher/student models.

Measures:
  - encoder_params_M   : vision encoder parameter count (M)
  - total_params_M     : total model parameter count (M)
  - mem_mib            : GPU peak memory during one forward pass (MiB)
  - speed_s_per_iter   : average wall-clock time per forward pass (seconds)
"""

import time
import torch


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(module):
    """Return total parameter count of a nn.Module."""
    return sum(p.numel() for p in module.parameters())


def get_encoder_module(model):
    """
    Return the vision encoder sub-module of a teacher or student model.

    Priority order:
      self.mae      → MAE teacher
      self.vit      → TinyViT student
      self.backbone → EfficientNet-B3/CLIP teacher or EfficientNet-B0/MobileNet-V3 student
      fallback      → whole model
    """
    m = model.module if hasattr(model, 'module') else model
    if hasattr(m, 'mae'):
        return m.mae
    if hasattr(m, 'vit'):
        return m.vit
    if hasattr(m, 'backbone'):
        return m.backbone
    return m


# ---------------------------------------------------------------------------
# Forward helper
# ---------------------------------------------------------------------------

def _run_forward(model, input_tensor):
    """Single forward pass compatible with teacher and student signatures."""
    m = model.module if hasattr(model, 'module') else model
    try:
        return m.forward(input_tensor)
    except TypeError:
        return m.forward(input_tensor, P=None)


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_model(model, input_tensor, device, warmup=3, repeats=20):
    """
    Profile a model's forward-pass speed and GPU memory.

    Args:
        model        : nn.Module already on device (teacher or student)
        input_tensor : sample input already on device — [1, C, H, W] or [1, K, C, H, W]
        device       : torch.device
        warmup       : warm-up iterations (not timed)
        repeats      : timed iterations for averaging

    Returns:
        dict:
          encoder_params_M  (float)  — vision encoder params in millions
          total_params_M    (float)  — total params in millions
          mem_mib           (float)  — peak GPU memory in MiB (0.0 on CPU)
          speed_s_per_iter  (float)  — avg seconds per forward pass
    """
    model.eval()
    encoder = get_encoder_module(model)

    encoder_params_M = count_params(encoder) / 1e6
    total_params_M   = count_params(model)   / 1e6

    # --- Memory profiling (GPU only) ---
    mem_mib = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _run_forward(model, input_tensor)
        torch.cuda.synchronize(device)
        mem_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # --- Speed profiling ---
    with torch.no_grad():
        for _ in range(warmup):
            _run_forward(model, input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _run_forward(model, input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    speed_s_per_iter = (time.perf_counter() - t0) / repeats

    return {
        'encoder_params_M': round(encoder_params_M, 2),
        'total_params_M':   round(total_params_M,   2),
        'mem_mib':          round(mem_mib,           1),
        'speed_s_per_iter': round(speed_s_per_iter,  6),
    }


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def format_profile_line(label, vlm_type, hidden_size, profile):
    """Return a one-line summary string for metrics.txt."""
    speed_ms = profile['speed_s_per_iter'] * 1000
    return (
        f"  {label}: {vlm_type} | hidden={hidden_size}"
        f" | enc_params={profile['encoder_params_M']:.2f}M"
        f" | total={profile['total_params_M']:.2f}M"
        f" | mem={profile['mem_mib']:.0f}MiB"
        f" | speed={speed_ms:.2f}ms/iter"
    )
