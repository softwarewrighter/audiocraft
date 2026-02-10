# Text-to-SFX Generator Setup

Generate sound effects from text descriptions using Meta's AudioGen model.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- FFmpeg (install via `brew install ffmpeg` on macOS)

## Setup

### 1. Create and activate virtual environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

### 2. Install PyTorch

```bash
uv pip install torch torchaudio torchvision
```

### 3. Install dependencies

```bash
uv pip install \
    "av>=12.0.0" \
    einops \
    "flashy>=0.0.1" \
    "hydra-core>=1.1" \
    hydra_colorlog \
    julius \
    num2words \
    sentencepiece \
    spacy \
    huggingface_hub \
    tqdm \
    "transformers>=4.31.0" \
    demucs \
    librosa \
    soundfile \
    torchmetrics \
    encodec \
    protobuf \
    torchdiffeq
```

### 4. Install audiocraft

```bash
uv pip install --no-deps -e .
```

### 5. Create xformers stub (macOS only)

On macOS, xformers (which requires CUDA) isn't available. Create a stub module:

```bash
mkdir -p .venv/lib/python3.11/site-packages/xformers
```

Create `.venv/lib/python3.11/site-packages/xformers/__init__.py`:
```python
# Stub xformers module for macOS (no CUDA support)
```

Create `.venv/lib/python3.11/site-packages/xformers/ops.py`:
```python
# Stub xformers.ops module for macOS (no CUDA support)
import torch

class AttentionBias:
    pass

class LowerTriangularMask(AttentionBias):
    pass

def memory_efficient_attention(query, key, value, attn_mask=None, p=0.0):
    scale = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if p > 0.0:
        attn_weights = torch.dropout(attn_weights, p, training=True)
    return torch.matmul(attn_weights, value)

def unbind(tensor, dim=0):
    return torch.unbind(tensor, dim=dim)

class fmha:
    class Inputs:
        pass
    class BlockDiagonalMask:
        pass
    class BlockDiagonalCausalMask:
        pass
```

## Usage

```bash
# Basic usage (generates 5-second audio)
python text_to_sfx.py "dog barking"

# Specify duration and output file
python text_to_sfx.py "thunder and heavy rain" --duration 10 --output storm.wav

# Short form
python text_to_sfx.py "footsteps on gravel" -d 3 -o footsteps.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path | `output.wav` |
| `-d, --duration` | Duration in seconds | `5.0` |
| `--top-k` | Top-k sampling | `250` |
| `--top-p` | Top-p (nucleus) sampling | `0.0` |
| `--temperature` | Sampling temperature | `1.0` |
| `--cfg-coef` | Classifier-free guidance | `3.0` |

### Example Prompts

- `"dog barking in the distance"`
- `"thunder and heavy rain"`
- `"footsteps in a corridor"`
- `"car engine starting"`
- `"birds chirping in the morning"`
- `"ocean waves crashing on rocks"`
- `"fire crackling"`
- `"crowd cheering at a stadium"`

### Prompt Tips

- Be specific: `"large dog barking aggressively"` vs `"dog barking"`
- Include environment: `"footsteps in an empty hallway"`
- Mention qualities: `"soft rain"`, `"loud thunder"`, `"distant sirens"`

## Notes

- **First run**: Downloads the AudioGen model (~1.5GB) from Hugging Face
- **Device**: Automatically uses MPS (Apple Silicon), CUDA, or CPU
- **Output**: 16kHz mono WAV files
- **Model**: Uses `facebook/audiogen-medium` (1.5B parameters)
