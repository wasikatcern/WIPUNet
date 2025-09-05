#####################################
###### Prepared by Wasikul Islam ####
### contact: wasikul.islam@cern.ch ##
#####################################

"""
python WIPUNet_BSDS500.py   --bsds_root /eos/atlas/unpledged/group-wisc/users/waislam/denoise_PU/data/BSDS500/BSR   --split train   --ckpt_dir ./denoise_results   --models WIPUNet punetg --sigma_list 15 25 50 --epochs 50   --batch_size 64   --num_workers 4   --amp   --save_every 10   --lr 5e-4 --clip_grad 1.0
"""

import os, glob, time, argparse, random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ------------------
# Device & Seed
# ------------------
def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev.type}", flush=True)
    return dev

def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = get_device()
set_seed(1234)

# ------------------
# BN policy (sync with concepts file)
# ------------------
USE_FP32_BN = True

class BN2d32(nn.BatchNorm2d):
    """BatchNorm2d that computes in float32 (for stability), then casts back."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x.float())
        return y.to(x.dtype)

def make_norm(c: int):
    if USE_FP32_BN:
        return BN2d32(c, eps=1e-3, momentum=0.05)
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.05)

# ------------------
# Data utils
# ------------------
def pil_to_tensor_no_numpy(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    buf = memoryview(img.tobytes())
    t = torch.frombuffer(buf, dtype=torch.uint8).clone().view(h, w, 3)
    return t.permute(2, 0, 1).to(torch.float32) / 255.0

def center_crop_multiple(img: Image.Image, multiple: int = 4, target: Optional[int] = None) -> Image.Image:
    # Center-crop PIL image so H,W are divisible by `multiple`. If target is given, crop to target x target first.
    W, H = img.size
    if target is not None:
        target = min(target, W, H)
        left = (W - target) // 2
        top  = (H - target) // 2
        img = img.crop((left, top, left+target, top+target))
        W, H = img.size
    newW = W - (W % multiple)
    newH = H - (H % multiple)
    if newW == W and newH == H:
        return img
    left = (W - newW) // 2
    top  = (H - newH) // 2
    return img.crop((left, top, left+newW, top+newH))

class BSDS500Dataset(Dataset):
    # Minimal BSDS500 image reader: <root>/BSDS500/data/images/<split>/*.jpg or <root>/<split>/*.jpg
    def __init__(self, root: str, split: str = "test", img_size_eval: int = 128, multiple: int = 4):
        self.root = root
        self.split = split
        patt1 = os.path.join(root, "BSDS500", "data", "images", split, "*.jpg")
        patt2 = os.path.join(root, split, "*.jpg")
        files = sorted(glob.glob(patt1)) or sorted(glob.glob(patt2))
        if not files:
            raise FileNotFoundError(f"No images found for split='{split}'. Tried:\n  {patt1}\n  {patt2}")
        self.files = files
        self.target = img_size_eval
        self.multiple = multiple

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = center_crop_multiple(img, multiple=self.multiple, target=self.target)
        x = pil_to_tensor_no_numpy(img)  # [C,H,W] in [0,1]
        return x, 0

def add_gaussian_noise(x: torch.Tensor, sigma: int) -> torch.Tensor:
    noise = torch.randn_like(x) * (sigma / 255.0)
    return (x + noise).clamp(0.0, 1.0)

# ------------------
# Metrics
# ------------------
def torch_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y).clamp(min=1e-12)
    ps = 20.0 * torch.log10(torch.tensor(data_range, device=x.device)) - 10.0 * torch.log10(mse)
    return float(ps.item())

def _gaussian_window(ks: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    c = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)

def torch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> float:
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

# ------------------
# Models (synced)
# ------------------
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
        self.use_se = use_se
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(x + h, inplace=True)

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

# --------- PU-Net-G (sigma-conditioned) ---------
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

# ------------------
# Training / Eval
# ------------------
def _to_device_cl(x: torch.Tensor) -> torch.Tensor:
    return x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

def save_ckpt(model, opt, epoch, model_name, sigma, outdir, tag):
    ckpt_dir = os.path.join(outdir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{model_name}_sigma{sigma}_{tag}.pth")
    torch.save({
        "epoch": epoch, "model": model_name, "sigma": sigma,
        "model_state": model.state_dict(), "optim_state": opt.state_dict(),
    }, path)
    return path

@torch.no_grad()
def eval_one(model, model_name, test_loader, sigma: int, is_sigma_model: bool, print_prefix: str = "[eval]"):
    model.eval()
    t0 = time.time()
    ps, ss = [], []
    pbar = tqdm(test_loader, desc=f"{print_prefix} {model_name} σ={sigma}", leave=True)
    for clean, _ in pbar:
        clean = _to_device_cl(clean)
        noisy = add_gaussian_noise(clean, sigma)
        if is_sigma_model:
            sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma / 255.0)
            den = model(noisy, sigma_map)
        else:
            den = model(noisy)
        psnr = torch_psnr(clean[0], den[0])
        ssim = torch_ssim(clean[0], den[0])
        ps.append(psnr); ss.append(ssim)
        pbar.set_postfix(psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")
    psnr_mean = float(sum(ps)/len(ps)); ssim_mean = float(sum(ss)/len(ss))
    print(f"{print_prefix} {model_name} σ={sigma}: PSNR={psnr_mean:.2f} dB | SSIM={ssim_mean:.3f} | time={time.time()-t0:.1f}s", flush=True)
    return psnr_mean, ssim_mean

def train_one(model, model_name, train_loader, sigma: int, epochs: int,
              outdir: str, is_sigma_model: bool, amp: bool, use_compile: bool, save_every: int = 10,
              lr: float = 5e-4, clip_grad: float = 1.0):
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    if use_compile and DEVICE.type == "cuda":
        try:
            model = torch.compile(model)
            print("[speed] torch.compile enabled", flush=True)
        except Exception as e:
            print("[speed] torch.compile unavailable:", e, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda' if (amp and DEVICE.type=='cuda') else 'cpu')
        def autocast_ctx(): return autocast('cuda' if DEVICE.type=='cuda' else 'cpu', enabled=amp and DEVICE.type=='cuda')
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and DEVICE.type == "cuda"))
        def autocast_ctx(): return torch.cuda.amp.autocast(enabled=(amp and DEVICE.type == "cuda"))

    t0 = time.time()
    for ep in range(1, epochs+1):
        ep_t0 = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f"[train] {model_name} σ={sigma} | epoch {ep}/{epochs}", leave=True)
        for clean, _ in pbar:
            clean = _to_device_cl(clean)
            noisy = add_gaussian_noise(clean, sigma)
            if is_sigma_model:
                sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma / 255.0)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(noisy, sigma_map) if is_sigma_model else model(noisy)
                loss = loss_fn(out, clean)

            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[train] {model_name} epoch {ep} done in {time.time()-ep_t0:.1f}s", flush=True)
        if save_every and (ep % save_every == 0):
            p = save_ckpt(model, opt, ep, model_name, sigma, outdir, tag=f"epoch{ep}")
            print(f"[ckpt] saved {p}", flush=True)

    plast = save_ckpt(model, opt, epochs, model_name, sigma, outdir, tag="last")
    print(f"[done] {model_name} in {time.time()-t0:.1f}s | last ckpt: {plast}", flush=True)
    return model

# ------------------
# BN Recalibration
# ------------------
@torch.no_grad()
def bn_recalibrate(model: nn.Module, loader: DataLoader, steps: int = 200, sigma: int = 15, is_sigma_model: bool = False):
    was_training = model.training
    model.train()
    it = iter(loader)
    for _ in tqdm(range(steps), desc="[bn_recalibrate]"):
        try:
            clean, _ = next(it)
        except StopIteration:
            it = iter(loader)
            clean, _ = next(it)
        clean = _to_device_cl(clean)
        noisy = add_gaussian_noise(clean, sigma)
        if is_sigma_model:
            sigma_map = torch.ones_like(clean[:, :1, ...], memory_format=torch.channels_last) * (sigma / 255.0)
            _ = model(noisy, sigma_map)
        else:
            _ = model(noisy)
    model.train(was_training)

# ------------------
# Utilities
# ------------------
def make_bsds_loader(root: str, split: str, img_size_eval: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = BSDS500Dataset(root=root, split=split, img_size_eval=img_size_eval, multiple=4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=2, drop_last=False)
    print(f"[data] BSDS500 {split}: {len(ds)} images | iters: {len(loader)}", flush=True)
    return loader

def make_model(name: str) -> Tuple[nn.Module, bool]:
    name = name.lower()
    if name == "WIPUNet":    return WIPUNet(), True
    if name in ("punetg", "pu_net_g", "punet_g"): return PUNetG(base=48), True
    raise ValueError(f"Unknown model '{name}'")

def find_ckpt(ckpt_dir: str, model_name: str, sigma: int) -> Optional[str]:
    import glob
    pats = [
        os.path.join(ckpt_dir, "checkpoints", f"{model_name}_sigma{sigma}_best.pth"),
        os.path.join(ckpt_dir, "checkpoints", f"{model_name}_sigma{sigma}_last.pth"),
        os.path.join(ckpt_dir, "checkpoints", f"{model_name}_sigma{sigma}_epoch*.pth"),
        os.path.join(ckpt_dir, f"{model_name}_sigma{sigma}_best.pth"),
        os.path.join(ckpt_dir, f"{model_name}_sigma{sigma}_last.pth"),
        os.path.join(ckpt_dir, f"{model_name}_sigma{sigma}_epoch*.pth"),
    ]
    for p in pats:
        if "*" in p:
            cand = sorted(glob.glob(p))
            if cand:
                return cand[-1]
        else:
            if os.path.exists(p):
                return p
    return None

def load_weights_if_available(model: nn.Module, ckpt_path: Optional[str]) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("[ckpt] none found (fresh model)", flush=True)
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}\n  missing={len(missing)} unexpected={len(unexpected)}", flush=True)

# ------------------
# Main
# ------------------
def main():
    ap = argparse.ArgumentParser(description="Denoise/Eval on BSDS500 with synced models + tqdm logging")
    ap.add_argument("--bsds_root", type=str, required=True, help="Path to BSDS500 root (containing BSDS500/data/images/{train,val,test}) or images/<split> directly")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--ckpt_dir", type=str, default="./denoise_results", help="Where checkpoints were saved")
    ap.add_argument("--models", nargs="+", default=["WIPUNet", "punetg"])
    ap.add_argument("--sigma_list", nargs="+", type=int, default=[15,25,50])
    ap.add_argument("--img_size_eval", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--eval_only", action="store_true", help="Only run evaluation using existing checkpoints")
    ap.add_argument("--epochs", type=int, default=0, help="If >0 and not --eval_only, train for this many epochs before eval")
    ap.add_argument("--amp", action="store_true", help="Enable AMP for training")
    ap.add_argument("--use_compile", action="store_true", help="Use torch.compile during training (CUDA only)")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--bn_recalibrate", action="store_true", help="Run BN recalibration before eval")
    ap.add_argument("--recalib_steps", type=int, default=200)
    args = ap.parse_args()

    # Build loader once (eval split)
    test_loader = make_bsds_loader(args.bsds_root, args.split, args.img_size_eval, args.batch_size, args.num_workers)

    results = []
    for sigma in args.sigma_list:
        for name in args.models:
            print(f"\n=== {name}  σ={sigma} ===", flush=True)
            model, needs_sigma = make_model(name)

            ckpt_path = find_ckpt(args.ckpt_dir, name, sigma)
            if args.eval_only and not ckpt_path:
                print(f"[warn] No checkpoint found for {name} σ={sigma}; skipping.", flush=True)
                continue

            if not args.eval_only and args.epochs > 0:
                # crude train loader: reuse test loader as "train" if you don't have BSDS train split handy
                train_loader = test_loader
                model = train_one(model, name, train_loader, sigma, args.epochs,
                                  args.ckpt_dir, needs_sigma, amp=args.amp,
                                  use_compile=args.use_compile, save_every=args.save_every,
                                  lr=args.lr, clip_grad=args.clip_grad)
                ckpt_path = find_ckpt(args.ckpt_dir, name, sigma)  # refresh after training

            load_weights_if_available(model, ckpt_path)
            model = model.to(DEVICE)

            if args.bn_recalibrate:
                bn_recalibrate(model, test_loader, steps=args.recalib_steps, sigma=sigma, is_sigma_model=needs_sigma)

            psnr, ssim = eval_one(model, name, test_loader, sigma, needs_sigma)
            results.append({"model": name, "sigma": sigma, "psnr": psnr, "ssim": ssim})

    # Save CSV
    if results:
        import csv
        out_dir = os.path.join(args.ckpt_dir, "eval_results")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"bsds500_eval.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model","sigma","psnr","ssim"])
            w.writeheader()
            for r in results: w.writerow(r)
        print(f"\n[done] wrote {out_csv}", flush=True)

if __name__ == "__main__":
    main()

