"""Minimal fixes needed to run the public BrainMosaic code (github.com/Erikaqvq/BrainMosaic_ICLR26, commit 398ac25)."""

for path in ("models/backbone.py", "models/matcher.py"):
    text = open(path).read()
    if "import torch.nn as nn" not in text and "from torch import nn" not in text:
        text = text.replace("import torch\n", "import torch\nimport torch.nn as nn\n", 1)
        open(path, "w").write(text)
text = open("models/ModernTCN.py").read()
open("models/ModernTCN.py", "w").write(
    text.replace("d_ffn = dmodel * ffn_ratio", "d_ffn = int(dmodel * ffn_ratio)")
)
