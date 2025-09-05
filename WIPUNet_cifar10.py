# --- Local GPU-aware PyTorch setup (no Colab/Drive) ---
import os, random, torch, shutil
from datetime import datetime
from pathlib import Path
import subprocess, shlex
import csv, time
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image

# 0) CUDA allocator tweak (helps fragmentation on long runs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 1) Output folders (cross-platform)
OUTDIR = str(Path("./denoise_results").resolve())
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
print(f"[setup] OUTDIR = {OUTDIR}")

# 2) Environment info
print(f"[setup] PyTorch: {torch.__version__}")

# Try to show GPUs (on Windows/Linux). It's fine if nvidia-smi isn't present.
try:
    proc = subprocess.run(shlex.split("nvidia-smi -L"), capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        print("[setup] Detected GPU(s):")
        print(proc.stdout.strip())
except Exception:
    pass

# 3) Device selection
cuda_available = torch.cuda.is_available()
mps_available  = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
if cuda_available:
    DEVICE = torch.device("cuda")
elif mps_available:   # Apple Silicon fallback (works with your code paths; AMP will be off)
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[setup] CUDA available: {cuda_available}")
if DEVICE.type == "cuda":
    print("[setup] GPU:", torch.cuda.get_device_name(0))
    # Speed knobs for NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
elif DEVICE.type == "mps":
    print("[setup] Using Apple MPS backend")
else:
    print("[setup] CPU detected")

print("[setup] DEVICE:", DEVICE)

# 4) Repro
def set_seed(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
set_seed(1234)

# 5) Default run config (tweak freely)
#    NOTE: Other cells read these keys, so keep names consistent.
def _auto_workers():
    try:
        # half your cores (at least 2, at most 8) is a good default for CIFAR-10
        return max(2, min(8, (os.cpu_count() or 4) // 2))
    except Exception:
        return 2

CFG = dict(
    sigma=100, #25, #15,
    epochs=100,
    batch_size=(512 if DEVICE.type in ("cuda","mps") else 64),  # adjust per-model if OOM
    num_workers=_auto_workers(),
    amp=bool(DEVICE.type == "cuda"),         # AMP only on CUDA in this notebook
    save_best_only=True,                     # kept for compatibility
    use_compile=bool(DEVICE.type == "cuda"), # set False if you hit OOM/instability
    save_every=10,

    # Trainer knobs used in your train_one(...)
    lr=5e-4,
    clip_grad=1.0,
    bn_calibrate_steps=200
)

print("[setup] Effective CFG:", CFG)

# 6) Optional: require GPU (uncomment to hard-stop on CPU)
# REQUIRE_GPU = True
# if REQUIRE_GPU and DEVICE.type != "cuda":
#     raise SystemExit("No CUDA GPU detected. Install a CUDA-enabled PyTorch and try again.")


# ---------- NumPy-safe data ----------
def pil_to_tensor_no_numpy(img):
    if img.mode != "RGB": img = img.convert("RGB")
    w, h = img.size
    try:
        buf = memoryview(img.tobytes())
        t = torch.frombuffer(buf, dtype=torch.uint8).clone().view(h, w, 3)  # ← clone()
    except Exception:
        t = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(h, w, 3)
    return t.permute(2, 0, 1).to(torch.float32) / 255.0

class CIFARNoNumpy(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = pil_to_tensor_no_numpy(img)
        return img, target

def add_gaussian_noise(x: torch.Tensor, sigma: int) -> torch.Tensor:
    noise = torch.randn_like(x) * (sigma / 255.0)
    return (x + noise).clamp(0.0, 1.0)

# ---------- Metrics (torch-only) ----------
def torch_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y).clamp(min=1e-12)
    ps = 20.0 * torch.log10(torch.tensor(data_range, device=x.device)) - 10.0 * torch.log10(mse)
    return float(ps.item())

def _gaussian_window(ks: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    c = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)

def torch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
               window_size: int = 11, sigma: float = 1.5) -> float:
    C, H, W = x.shape
    win = _gaussian_window(window_size, sigma, device=x.device, dtype=x.dtype)
    pad = window_size // 2
    x = x.unsqueeze(0); y = y.unsqueeze(0)
    win_c = win.expand(C, 1, window_size, window_size)
    mu_x = F.conv2d(x, win_c, padding=pad, groups=C)
    mu_y = F.conv2d(y, win_c, padding=pad, groups=C)
    mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, win_c, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win_c, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win_c, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
               (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().clamp(0, 1).item())


# --------- Optional: BN in fp32 for stability under AMP ---------
USE_FP32_BN = True  # set False to use plain BatchNorm2d with safer eps/momentum

class BN2d32(nn.BatchNorm2d):
    """BatchNorm2d that computes in float32, returns input dtype."""
    def forward(self, x):
        y = super().forward(x.float())
        return y.to(x.dtype)

def make_norm(c: int):
    if USE_FP32_BN:
        return BN2d32(c, eps=1e-3, momentum=0.05)
    else:
        return nn.BatchNorm2d(c, eps=1e-3, momentum=0.05)

# --------- Shared blocks ---------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch//r, 1)
        self.fc2 = nn.Conv2d(ch//r, ch, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResBlock(nn.Module):
    def __init__(self, c, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1   = make_norm(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2   = make_norm(c)
        self.se    = SEBlock(c) if use_se else nn.Identity()
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(x + h, inplace=True)

# --------- Baselines ---------
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding),
                       make_norm(n_channels), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return (x - self.net(x)).clamp(0, 1)

class FFDNet(nn.Module):
    def __init__(self, n_channels=64, image_channels=3):
        super().__init__()
        self.head = nn.Conv2d(image_channels + 1, n_channels, 3, padding=1)
        body = []
        for _ in range(7):
            body += [nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(n_channels, image_channels, 3, padding=1)
    def forward(self, x, sigma_map):
        h = torch.cat([x, sigma_map], dim=1)
        h = F.relu(self.head(h), inplace=True)
        h = self.body(h)
        noise = self.tail(h)
        return (x - noise).clamp(0, 1)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.enc1, self.enc2 = UNetBlock(image_channels, 64), UNetBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec1, self.dec2 = UNetBlock(128, 64), UNetBlock(64, 64)
        self.final = nn.Conv2d(64, image_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = F.interpolate(e2, scale_factor=2, mode="nearest")
        d = self.dec1(d)
        d = self.dec2(d + e1)
        return (x - self.final(d)).clamp(0, 1)

class RestormerLite(nn.Module):
    def __init__(self, image_channels=3, embed_dim=48, num_heads=4, depth=2):
        super().__init__()
        self.proj = nn.Conv2d(image_channels, embed_dim, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.ReLU(), nn.Linear(embed_dim*2, embed_dim))
            ]) for _ in range(depth)
        ])
        self.out = nn.Conv2d(embed_dim, image_channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.proj(x).flatten(2).transpose(1, 2)  # [B, HW, C]
        for attn, mlp in self.blocks:
            a, _ = attn(h, h, h, need_weights=False)  # avoid computing attn weights
            h = h + a
            h = h + mlp(h)
        h = h.transpose(1, 2).reshape(B, -1, H, W)
        noise = self.out(h)
        return (x - noise).clamp(0, 1)

# --------- PU-Net++ (core & wrapper) ---------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.seq(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2)
        self.down2 = nn.MaxPool2d(2)
        self.bott = ConvBlock(base*2, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        xb = self.bott(self.down2(x2))
        x  = self.up2(xb)
        x  = self.dec2(torch.cat([x, x2], dim=1))
        x  = self.up1(x)
        x  = self.dec1(torch.cat([x, x1], dim=1))
        return x

class ConservativeSplit(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, Y, m, g=None):
        if g is None: g = 1.0
        m = torch.clamp(m, 0.0, 1.0)
        S = (g * m) * Y
        B = Y - S
        return S, B

class PUNetPPCore(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.backbone = UNetSmall(in_ch=3, base=base)
        self.density  = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, 1, 3, padding=1), nn.Softplus()
        )
        self.mask     = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, 1, 3, padding=1), nn.Sigmoid()
        )
        self.gate     = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, 1, 3, padding=1), nn.Sigmoid()
        )
        self.splitter = ConservativeSplit()
    def forward(self, Y):
        F   = self.backbone(Y)
        rho = self.density(F)
        m   = self.mask(F)
        g   = self.gate(F)
        residual = (Y - rho).clamp(0.0, 1.0)
        S_hat = (g * m) * residual
        B_hat = Y - S_hat
        return S_hat, B_hat, {"rho": rho, "m": m, "g": g, "residual": residual}

class PUNetPP(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.core = PUNetPPCore(base=base)
    def forward(self, x):
        S_hat, _, _ = self.core(x)
        return S_hat.clamp(0, 1)

# --------- PU-Net-G (sigma conditioned) ---------
class PUGaussDown(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 2, 1),
            nn.ReLU(True),
            ResBlock(c_out, use_se=False),
            ResBlock(c_out, use_se=False),
        )
    def forward(self, x): return self.net(x)

class PUGaussUp(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, 2)
        self.post = nn.Sequential(
            ResBlock(c_out*2, use_se=False),
            ResBlock(c_out*2, use_se=False),
            nn.Conv2d(c_out*2, c_out, 3, 1, 1),
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = self.post(torch.cat([x, skip], dim=1))
        return x

class PUGaussNet(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        c0 = base
        self.head = nn.Sequential(
            nn.Conv2d(4, c0, 3, 1, 1), nn.ReLU(True),
            ResBlock(c0, use_se=False),
        )
        self.d1 = PUGaussDown(c0,   c0*2)
        self.d2 = PUGaussDown(c0*2, c0*4)
        self.bott = nn.Sequential(
            ResBlock(c0*4, use_se=False),
            ResBlock(c0*4, use_se=False),
        )
        self.u2 = PUGaussUp(c0*4, c0*2)
        self.u1 = PUGaussUp(c0*2, c0)
        self.tail = nn.Sequential(
            ResBlock(c0, use_se=False),
            nn.Conv2d(c0, 3, 3, 1, 1)
        )
    def forward(self, Y, sigma_map):
        h0 = self.head(torch.cat([Y, sigma_map], dim=1))
        s1 = h0
        h1 = self.d1(h0); s2 = h1
        h2 = self.d2(h1)
        hb = self.bott(h2)
        u2 = self.u2(hb, s2)
        u1 = self.u1(u2, s1)
        N_hat = self.tail(u1)
        S_hat = (Y - N_hat).clamp(0, 1)
        return S_hat, N_hat

class PUNetG(nn.Module):
    def __init__(self, base=48):
        super().__init__()
        self.core = PUGaussNet(base=base)
    def forward(self, x, sigma_map):
        S_hat, _ = self.core(x, sigma_map)
        return S_hat

# --------- WIPUNet configurable variants ---------
class DownLearned(nn.Module):
    def __init__(self, c_in, c_out, use_se=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 2, 1),
            nn.ReLU(True),
            ResBlock(c_out, use_se), ResBlock(c_out, use_se)
        )
    def forward(self, x): return self.net(x)

class UpLearned(nn.Module):
    def __init__(self, c_in, c_out, use_se=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, 2)
        self.post = nn.Sequential(
            ResBlock(c_out*2, use_se), ResBlock(c_out*2, use_se),
            nn.Conv2d(c_out*2, c_out, 3, 1, 1)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = self.post(torch.cat([x, skip], dim=1))
        return x

class DownFixed(nn.Module):
    def __init__(self, c_in, c_out, use_se=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.post = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.ReLU(True),
            ResBlock(c_out, use_se), ResBlock(c_out, use_se)
        )
    def forward(self, x):
        return self.post(self.pool(x))

class UpFixed(nn.Module):
    def __init__(self, c_in, c_out, use_se=True):
        super().__init__()
        ch = c_in + c_out
        self.post = nn.Sequential(
            ResBlock(ch, use_se),
            ResBlock(ch, use_se),
            nn.Conv2d(ch, c_out, 3, 1, 1)
        )
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.post(x)

class WIPUNetConfigurable(nn.Module):
    def __init__(self, base=64, hard_conservation=True, noise_conditioning=True, use_se=True, learned_resampling=True):
        super().__init__()
        self.hard_conservation = hard_conservation
        self.noise_conditioning = noise_conditioning
        self.use_se = use_se
        self.learned_resampling = learned_resampling
        in_ch = 3 + (1 if self.noise_conditioning else 0)
        c0 = base
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, 1, 1), nn.ReLU(True),
            ResBlock(c0, use_se=self.use_se)
        )
        Down = DownLearned if self.learned_resampling else DownFixed
        Up   = UpLearned   if self.learned_resampling else UpFixed
        self.enc2 = Down(c0, c0*2, use_se=self.use_se)
        self.enc3 = Down(c0*2, c0*4, use_se=self.use_se)
        self.bott = nn.Sequential(
            ResBlock(c0*4, use_se=self.use_se),
            (SEBlock(c0*4) if self.use_se else nn.Identity()),
            ResBlock(c0*4, use_se=self.use_se),
        )
        self.dec3 = Up(c0*4, c0*2, use_se=self.use_se)
        self.dec2 = Up(c0*2, c0,   use_se=self.use_se)
        self.tail = nn.Conv2d(c0, 3, 3, 1, 1)
    def forward(self, x, sigma_map=None):
        h = torch.cat([x, sigma_map], dim=1) if self.noise_conditioning else x
        h1 = self.head(h)
        e2 = self.enc2(h1)
        e3 = self.enc3(e2)
        b  = self.bott(e3)
        d3 = self.dec3(b, e2)
        d2 = self.dec2(d3, h1)
        out = self.tail(d2)
        if self.hard_conservation:
            return (x - out).clamp(0, 1)
        return out.clamp(0, 1)

class WIPUNet(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, True, True, True, True)
# Named variants
class WIPUNet1(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, True, False, False, False)
class WIPUNet2(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, True, False, False)
class WIPUNet3(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, False, True, False)
class WIPUNet4(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, False, False, True)
class WIPUNet_123(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, True, True, True, False)
class WIPUNet_124(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, True, True, False, True)
class WIPUNet_134(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, True, False, True, True)
class WIPUNet_234(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, True, True, True)
class WIPUNet_23(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, True, True, False)
class WIPUNet_24(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, True, False, True)
class WIPUNet_34(WIPUNetConfigurable):
    def __init__(self, base=64): super().__init__(base, False, False, True, True)

# ----------------
# Data loaders
# ----------------
def make_loaders(batch_size=128, num_workers=2, data_root="./data"):
    """
    Provide your dataset classes externally. Example below assumes CIFARNoNumpy.
    """
    from torch.utils.data import DataLoader  # ensure import exists in your long script

    # Users: swap with your dataset(s) as needed.
    trainset = CIFARNoNumpy(root=data_root, train=True, download=True)
    testset  = CIFARNoNumpy(root=data_root, train=False, download=True)

    # Only enable pinned memory/persistent workers when CUDA & workers>0
    use_cuda = (DEVICE.type == "cuda")
    common = dict(num_workers=max(0, int(num_workers)))
    if use_cuda:
        common.update(pin_memory=True)
    if common["num_workers"] > 0:
        common.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    test_loader  = DataLoader(testset,  batch_size=min(64, batch_size), shuffle=False,
                              drop_last=False, **common)

    print(f"[data] train iters/epoch: {len(train_loader)} | test iters: {len(test_loader)}")
    return train_loader, test_loader

# ----------------
# Checkpointing
# ----------------
def save_ckpt(model, opt, epoch, model_name, sigma, outdir, tag):
    ckpt_dir = os.path.join(outdir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{model_name}_sigma{sigma}_{tag}.pth")
    torch.save({
        "epoch": epoch,
        "model": model_name,
        "sigma": sigma,
        "model_state": model.state_dict(),
        "optim_state": (opt.state_dict() if opt is not None else None),
    }, path)
    return path

# ----------------
# Eval helpers
# ----------------
def _to_device_cl(x):  # channels_last only helpful on CUDA tensors
    x = x.to(DEVICE, non_blocking=True)
    if DEVICE.type == "cuda":
        x = x.to(memory_format=torch.channels_last)
    return x

@torch.no_grad()
def eval_one(model, model_name, test_loader, sigma: int, is_sigma_model: bool):
    model.eval()
    ps_vals, ss_vals = [], []

    for clean, _ in tqdm(test_loader, desc=f"[eval] {model_name} σ={sigma}"):
        clean = _to_device_cl(clean)
        noisy = add_gaussian_noise(clean, sigma)  # user-provided

        if is_sigma_model:
            sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma / 255.0) \
                        if DEVICE.type == "cuda" else torch.ones_like(clean[:, :1, ...]) * (sigma / 255.0)
            den = model(noisy, sigma_map)
        else:
            den = model(noisy)

        # Compute PSNR/SSIM per-sample; average minibatch -> list
        bsz = clean.shape[0]
        for i in range(bsz):
            ps_vals.append(float(torch_psnr(clean[i], den[i])))   # user-provided
            ss_vals.append(float(torch_ssim(clean[i], den[i])))   # user-provided

    # Global averages
    ps = sum(ps_vals) / max(1, len(ps_vals))
    ss = sum(ss_vals) / max(1, len(ss_vals))
    return ps, ss

# ----------------
# BN utilities (prevents NaNs at eval)
# ----------------
def bn_with_bad_stats(model):
    bad = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            rm, rv = m.running_mean, m.running_var
            if (rm is not None and (torch.isnan(rm).any() or torch.isinf(rm).any())) \
               or (rv is not None and (torch.isnan(rv).any() or torch.isinf(rv).any())):
                bad.append(name)
    return bad

def reset_bn_running_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.running_mean is not None: m.running_mean.zero_()
            if m.running_var  is not None: m.running_var.fill_(1.0)

@torch.no_grad()
def recalibrate_bn(model, loader, sigma, is_sigma_model, steps=200):
    """Refresh BN running stats from fresh data (no grad)."""
    model.train()
    it = 0
    for clean, _ in loader:
        clean = _to_device_cl(clean)
        noisy = add_gaussian_noise(clean, sigma)
        if is_sigma_model:
            sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma/255.0) \
                        if DEVICE.type == "cuda" else torch.ones_like(clean[:, :1, ...]) * (sigma/255.0)
            _ = model(noisy, sigma_map)
        else:
            _ = model(noisy)
        it += 1
        if it >= steps:
            break
    model.eval()

# ----------------
# Trainer (AMP + clipping + optional compile + BN recalib)
# ----------------
def train_one(model, model_name, train_loader, sigma: int, epochs: int,
              outdir: str, is_sigma_model: bool, amp: bool, use_compile: bool,
              save_every: int = 10, lr: float = 5e-4, clip_grad: float = 1.0,
              bn_calibrate_steps: int = 200):
    """
    Minimal external deps:
      - add_gaussian_noise(tensor, sigma)
      - model forward: model(noisy) or model(noisy, sigma_map)
    """
    # Move model
    if DEVICE.type == "cuda":
        model = model.to(DEVICE).to(memory_format=torch.channels_last)
    else:
        model = model.to(DEVICE)

    # Optional compile (PyTorch 2.x+)
    if use_compile and DEVICE.type == "cuda":
        try:
            model = torch.compile(model)
            print("[speed] torch.compile enabled")
        except Exception as e:
            print("[speed] torch.compile unavailable:", e)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=(amp and DEVICE.type == "cuda"))

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[train] {model_name} σ={sigma} | epoch {ep}/{epochs}", leave=False)
        for clean, _ in pbar:
            clean = _to_device_cl(clean)
            noisy = add_gaussian_noise(clean, sigma)

            sigma_map = None
            if is_sigma_model:
                sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma / 255.0) \
                            if DEVICE.type == "cuda" else torch.ones_like(clean[:, :1, ...]) * (sigma / 255.0)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(amp and DEVICE.type == "cuda")):
                out  = model(noisy, sigma_map) if is_sigma_model else model(noisy)
                loss = loss_fn(out, clean)

            # guard against bad loss
            if not torch.isfinite(loss):
                print("[warn] non-finite loss; skipping step")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # needed before clipping
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(opt)
            scaler.update()

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if save_every and (ep % save_every == 0):
            p = save_ckpt(model, opt, ep, model_name, sigma, outdir, tag=f"epoch{ep}")
            print(f"[ckpt] saved {p}")

    # Auto BN recalibration → prevents NaNs at eval
    if bn_calibrate_steps and bn_calibrate_steps > 0:
        bad = bn_with_bad_stats(model)
        if bad:
            print("[bn] found bad running stats in:", bad)
        print(f"[bn] recalibrating BN for {bn_calibrate_steps} mini-batches…")
        recalibrate_bn(model, train_loader, sigma, is_sigma_model, steps=bn_calibrate_steps)

    plast = save_ckpt(model, opt, epochs, model_name, sigma, outdir, tag="last")
    print(f"[done] {model_name} in {time.time()-t0:.1f}s | last ckpt: {plast}")
    return model

# ----------------
# Model factory (leave as-is; you provide the classes)
# ----------------
def make_model(name: str):
    name = name.lower()
    if name == "dncnn":           return DnCNN(), False
    if name == "ffdnet":          return FFDNet(), True
    if name == "unet":            return UNet(), False
    if name == "restormer":       return RestormerLite(), False
    if name == "punetpp":         return PUNetPP(base=32), False
    if name in ("punetg","pu_net_g","punet_g"): return PUNetG(base=48), True
    if name == "WIPUNet":    return WIPUNet(), True
    raise ValueError(f"Unknown model '{name}'")

##########################################################
# --- Run WIPUNet (speed-optimized, memory-safe) ---

# Reduce CUDA fragmentation (set before CUDA work)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# 1) Try a larger batch if VRAM allows; drop if you OOM
pu_batch = min(CFG["batch_size"], 192)  # 192 often fits on T4 for base=48
train_loader, test_loader = make_loaders(pu_batch, CFG["num_workers"])

# 2) Model width: keep small for speed; raise to 64 only if you want more quality
model_base = 48   # try 40 or 32 for even faster runs (with some PSNR drop)
model = WIPUNet(base=model_base)
needs_sigma = True

# 3) Compile only when likely to fit (small batch); otherwise False to save VRAM
use_compile = bool(CFG.get("use_compile", False) and pu_batch <= 256)

# 4) Train (AMP on), with fewer BN calibration steps and no periodic ckpts
model = train_one(
    model, "WIPUNet",
    train_loader, CFG["sigma"], CFG["epochs"],
    OUTDIR, needs_sigma,
    amp=CFG["amp"],
    use_compile=use_compile,                # compile if small batch fits
    save_every=0,                           # skip periodic checkpoints (I/O speedup)
    lr=CFG.get("lr", 5e-4),
    clip_grad=CFG.get("clip_grad", 1.0),
    bn_calibrate_steps=64                   # <- was 200; 64 is usually plenty
)

# 5) Eval
if torch.cuda.is_available():
    torch.cuda.empty_cache()
psnr, ssim = eval_one(model, "WIPUNet", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] WIPUNet σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

# 6) Save a recalibrated checkpoint (stable BN stats for future evals)
ckpt_dir = os.path.join(OUTDIR, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
recalib_path = os.path.join(ckpt_dir, f"WIPUNet_sigma{CFG['sigma']}_last_recalib.pth")
torch.save({"model":"WIPUNet","sigma":CFG["sigma"],"model_state":model.state_dict()}, recalib_path)
print("[save] recalibrated ckpt ->", recalib_path)

##########################################################


'''
##########################################################
### --- Run dncnn ###
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("dncnn")
model = train_one(model, "dncnn", train_loader, CFG["sigma"], CFG["epochs"],
                  OUTDIR, needs_sigma, CFG["amp"], CFG["use_compile"], save_every=CFG["save_every"])
psnr, ssim = eval_one(model, "dncnn", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] DnCNN σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

##########################################################
### --- Run ffdnet ###
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("ffdnet")
model = train_one(model, "ffdnet", train_loader, CFG["sigma"], CFG["epochs"],
                  OUTDIR, needs_sigma, CFG["amp"], CFG["use_compile"], save_every=CFG["save_every"])
psnr, ssim = eval_one(model, "ffdnet", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] FFDNet σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

##########################################################
### --- Run unet ###
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("unet")
model = train_one(model, "unet", train_loader, CFG["sigma"], CFG["epochs"],
                  OUTDIR, needs_sigma, CFG["amp"], CFG["use_compile"], save_every=CFG["save_every"])
psnr, ssim = eval_one(model, "unet", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] UNet σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

##########################################################

# --- FAST RestormerLite run (smaller model + mem-efficient attention) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Prefer mem-efficient scaled dot-product attention (good on T4)
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(False)        # flash needs Ampere (T4 won't use it)
    sdp_kernel.enable_math_sdp(False)         # avoid slow math fallback
    sdp_kernel.enable_mem_efficient_sdp(True) # Triton mem-efficient attention
    print("[attn] mem-efficient SDPA enabled")
except Exception as e:
    print("[attn] SDPA config skipped:", e)

# Use a larger batch now that the model is lighter (fallback to 128 if OOM)
restormer_batch = 256

# Smaller Restormer is dramatically faster than 48x4-heads:
embed_dim = 32
num_heads = 2
depth = 2

train_loader, test_loader = make_loaders(restormer_batch, CFG["num_workers"])

# Instantiate the lighter model directly (skip make_model's default size)
fast_model = RestormerLite(embed_dim=embed_dim, num_heads=num_heads, depth=depth)
needs_sigma = False

# torch.compile can help now that the model is smaller; keep AMP on
model = train_one(
    fast_model, "restormer_fast",
    train_loader, CFG["sigma"], CFG["epochs"],
    OUTDIR, needs_sigma,
    amp=CFG["amp"],
    use_compile=True,                      # was False before; True usually speeds this up
    save_every=CFG.get("save_every", 10)
)

psnr, ssim = eval_one(model, "restormer_fast", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] RestormerFast σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

##########################################################
### --- Run punetpp ##
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("punetpp")
model = train_one(model, "punetpp", train_loader, CFG["sigma"], CFG["epochs"],
                  OUTDIR, needs_sigma, CFG["amp"], CFG["use_compile"], save_every=CFG["save_every"])
psnr, ssim = eval_one(model, "punetpp", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] PUNetPP σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")


##########################################################
### --- Run punetg ##
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("punetg")
model = train_one(model, "punetg", train_loader, CFG["sigma"], CFG["epochs"],
                  OUTDIR, needs_sigma, CFG["amp"], CFG["use_compile"], save_every=CFG["save_every"])
psnr, ssim = eval_one(model, "punetg", test_loader, CFG["sigma"], needs_sigma)
print(f"[test] PUNetG σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")

'''


'''
##########################################################
### Evaluate with a specific saved checkpoint 
##########################################################

# --- Recreate data + model
train_loader, test_loader = make_loaders(CFG["batch_size"], CFG["num_workers"])
model, needs_sigma = make_model("punetg")
model = model.to(DEVICE)

# --- Checkpoint to test
ckpt_path = "/eos/atlas/unpledged/group-wisc/users/waislam/denoise_PU/denoise_results/checkpoints/punetg_sigma15_epoch50.pth"
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

# --- Load & sanitize state_dict (handles torch.compile/_orig_mod and DataParallel/module)
def _sanitize(sd):
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd

ckpt = torch.load(ckpt_path, map_location=DEVICE)
state = ckpt.get("model_state", ckpt)  # support raw state_dict or wrapped
state = _sanitize(state)

missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[warn] load_state_dict mismatches:")
    if missing:   print("  missing:",   missing)
    if unexpected:print("  unexpected:", unexpected)

print(f"[ckpt] Loaded {ckpt_path} (epoch {ckpt.get('epoch','?')})")

# --- (Optional) BN recalibration to avoid NaNs / poor eval due to stale running stats
# Only do this if you see bad BN stats or eval is unstable.
# recalibrate_bn(model, train_loader, sigma=CFG["sigma"], is_sigma_model=needs_sigma, steps=CFG.get("bn_calibrate_steps", 200))

# --- Evaluate
psnr, ssim = eval_one(model, "punetg", test_loader, CFG["sigma"], needs_sigma)
print(f"[test @epoch{ckpt.get('epoch','?')}] PUNetG σ={CFG['sigma']}: PSNR={psnr:.2f} dB | SSIM={ssim:.3f}")
'''

##########################################################
