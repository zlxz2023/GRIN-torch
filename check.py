import torch
import numpy as np
import sys
import os

# ===== 路径对齐（和你 run_experiment 一样）=====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
from lib.nn import models
from lib import fillers, datasets
import warnings
warnings.filterwarnings("ignore")
# ========= 1. 固定随机种子 =========
torch.manual_seed(42)
np.random.seed(42)

# ========= 2. 构造 dataset（保证 N & adj 一致） =========
dataset = datasets.GuangzhouAirQuality(
    impute_nans=True,
    excluded_sensors=[30, 28, 1, 31, 6, 52, 51, 2, 35, 27, 39, 23, 46, 3],
    val_ratio=0.1,
    test_ratio=0.2
)

adj = dataset.get_similarity(thr=0.1)
np.fill_diagonal(adj, 0.0)
N = adj.shape[0]

# ========= 3. 构造 dummy 输入 =========
B, T, F = 2, 12, 1
x = torch.randn(B, T, N, F)
mask = torch.rand(B, T, N, F) > 0.3

# ========= 4. 构造 model kwargs（⚠ 必须与 PL 完全一致） =========
model_kwargs = dict(
    adj=adj,
    d_in=F,
    d_hidden=64,
    d_ff=64,
    ff_dropout=0.0,
    n_layers=1,
    kernel_size=2,
    decoder_order=1,
    global_att=False,
    d_u=0,
    d_emb=8,
    layer_norm=False,
    merge="mlp",
    impute_only_holes=False,
)

# ========= 5. 构造 Torch-only filler =========
filler = fillers.GraphFiller(
    model_class=models.GRINet,
    model_kwargs=model_kwargs,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.01, 'weight_decay': 0.1},
    loss_fn=None,
    metrics=None,
)

# ========= 6. load PL ckpt =========

ckpt = torch.load(
    "grin_full_model.pth",
    map_location="cpu"
)
filler.load_state_dict(ckpt["model_state_dict"], strict=True)
filler.eval()

# ========= 7. forward =========
with torch.no_grad():
    out = filler.model(x, mask)

print("✅ Torch-only forward 完成")
print("mean:", out.mean().item())
print("std :", out.std().item())