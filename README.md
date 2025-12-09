# Python environment setup

Create and activate a virtual environment, then install dependencies:

```bash
cd /Users/aniket/Projects/coherence_model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
```
