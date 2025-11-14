# train_synthetic_vibration.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# ----------------- Narrowband synthesizer + segmenter -----------------
def biquad_bandpass(fc, Q, fs):
    w0 = 2*np.pi*fc/fs
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)
    b0 =   Q*alpha
    b2 =  -Q*alpha
    a0 =   1 + alpha
    a1 =  -2*cosw0
    a2 =   1 - alpha
    b = np.array([b0/a0, 0.0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return b, a

def lfilter(b, a, x):
    y = np.zeros_like(x, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = b
    a0, a1, a2 = a
    for n in range(len(x)):
        xn = x[n]
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y

def fm_phase(fs, f0, N, sigma=0.001, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1.0/fs
    drift = np.cumsum(rng.normal(0, sigma, size=N)) / 50.0
    inst_f = f0 * (1.0 + drift)
    return 2*np.pi*np.cumsum(inst_f)*dt

def bandlimited_noise(N, fs, fc, Q, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, size=N)
    b, a = biquad_bandpass(fc, Q, fs)
    return lfilter(b, a, noise)

CLASSES = ["normal", "imbalance", "misalignment", "inner_race", "outer_race", "ball_fault"]

def synth_sample(fs=12000, N=4096, f_rot=30.0, f_res=3000.0, Q=35.0,
                 mode="normal", snr_db=25.0, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(N)/fs
    b, a = biquad_bandpass(f_res, Q, fs)

    # small 1×/2× content passed through the resonance path
    ph1 = fm_phase(fs, f_rot,   N, 0.0015, rng.integers(1e9))
    ph2 = fm_phase(fs, 2*f_rot, N, 0.0015, rng.integers(1e9))
    base = 0.02*np.sin(ph1) + 0.01*np.sin(ph2)
    x = lfilter(b, a, base)

    if mode == "imbalance":
        ph = fm_phase(fs, f_rot, N, 0.003, rng.integers(1e9))  # More frequency variation
        x += lfilter(b, a, 0.06*np.sin(ph))  # Reduced amplitude (50% of original)
    elif mode == "misalignment":
        ph = fm_phase(fs, 2*f_rot, N, 0.003, rng.integers(1e9))  # More frequency variation
        x += lfilter(b, a, 0.045*np.sin(ph))  # Reduced amplitude (50% of original)
    elif mode in ("inner_race", "outer_race", "ball_fault"):
        mult = {"inner_race": 5.2, "outer_race": 3.1, "ball_fault": 4.7}[mode]
        f_bp = mult * f_rot
        period = int(round(fs/f_bp))
        idx = 0
        while idx < N:
            pos = idx + period + rng.integers(-4, 5)  # More jitter in timing
            if pos >= N: break
            imp = np.zeros(N); imp[pos] = rng.uniform(0.4, 0.8)  # Reduced impact amplitude
            x += lfilter(b, a, imp)
            idx = pos
        x *= (1.0 + 0.05*np.sin(2*np.pi*1.2*t + rng.uniform(0, 2*np.pi)))  # Reduced AM

    # band-limited noise to keep spectrum narrow
    nb = bandlimited_noise(N, fs, f_res, Q, rng.integers(1e9))
    
    # Add additional broadband noise for more challenging detection
    broadband_noise = rng.normal(0, 0.01, size=N)  # Additional white noise
    
    # Add low-frequency drift (simulating measurement variations)
    drift = 0.02 * np.sin(2*np.pi*0.1*t + rng.uniform(0, 2*np.pi))
    
    sig_pow = np.mean(x**2) + 1e-12
    snr_lin = 10**(snr_db/10.0)
    nb = nb / (np.std(nb) + 1e-12) * np.sqrt(sig_pow/snr_lin)
    x = x + nb + broadband_noise + drift
    x *= (1.0 + 0.03*np.sin(2*np.pi*0.4*t + rng.uniform(0, 2*np.pi)))  # Reduced overall AM
    
    # Normalize amplitude to ensure similar y-range across all classes
    # Target RMS value for consistent amplitude scaling
    target_rms = 0.02  # Target RMS value for all classes
    current_rms = np.sqrt(np.mean(x**2)) + 1e-12
    x = x * (target_rms / current_rms)
    
    return x.astype(np.float32)

def make_segmented_dataset(
    fs=12000, run_duration_s=6, segment_len=2048, hop=1024,
    classes=CLASSES, f_res=3000.0, Q=35.0, snr_db=25.0, seed=0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    N_run = int(run_duration_s*fs)
    X_, y_ = [], []
    for li, cls in enumerate(classes):
        sig = synth_sample(fs, N_run, 30.0, f_res, Q, cls, snr_db, seed=rng.integers(1e9))
        for start in range(0, N_run - segment_len + 1, hop):
            seg = sig[start:start+segment_len]
            X_.append(seg[None, :])     # (1, L) univariate
            y_.append(li)
    X = np.stack(X_).astype(np.float32)  # (N, 1, L)
    y = np.array(y_, dtype=np.int64)
    
    # Additional per-class amplitude normalization to ensure similar y-ranges
    for class_idx in range(len(classes)):
        class_mask = y == class_idx
        class_data = X[class_mask]
        
        if len(class_data) > 0:
            # Normalize each class to have similar amplitude statistics
            class_rms = np.sqrt(np.mean(class_data**2, axis=(1, 2), keepdims=True))
            target_class_rms = 0.02  # Target RMS for all classes
            X[class_mask] = class_data * (target_class_rms / (class_rms + 1e-12))
    
    # shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm], list(classes)

# ----------------- PyTorch dataset + model -----------------
class VibSegments(Dataset):
    def __init__(self, X, y, zscore=True):
        self.X = X
        self.y = y
        self.z = zscore
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.z:
            m = x.mean(axis=-1, keepdims=True); s = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - m)/s
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

class Tiny1DCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 16,  kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=9, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        h = self.feat(x).squeeze(-1)
        return self.fc(h)

# ----------------- Train / evaluate -----------------
def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()

def confusion_matrix(pred, target, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred, target):
        cm[t, p] += 1
    return cm

def main():
    # 1) Generate segmented data
    fs = 12000
    X, y, names = make_segmented_dataset(
        fs=fs, run_duration_s=8, segment_len=2048, hop=1024,
        classes=["normal","imbalance","misalignment","inner_race","outer_race","ball_fault"],
        f_res=3000.0, Q=40.0, snr_db=28.0, seed=7
    )
    n_classes = len(names)
    print("Data:", X.shape, y.shape, names)

    # 2) Split (stratified by simple slicing per class)
    # here we just do an 80/20 split over the shuffled set
    n = len(X); split = int(0.8*n)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # 3) Dataloaders
    train_ds = VibSegments(X_train, y_train, zscore=True)
    test_ds  = VibSegments(X_test,  y_test,  zscore=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    # 4) Model/optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tiny1DCNN(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 5) Train
    for epoch in range(10):
        model.train()
        loss_sum = 0.0; acc_sum = 0.0; cnt = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            acc_sum  += (logits.argmax(1) == yb).float().sum().item()
            cnt      += xb.size(0)
        print(f"Epoch {epoch+1:02d} | train loss {loss_sum/cnt:.4f} acc {acc_sum/cnt:.3f}")

    # 6) Evaluate
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_pred.append(logits.cpu())
            all_true.append(yb)
    y_pred = torch.cat(all_pred).argmax(1).numpy()
    y_true = torch.cat(all_true).numpy()
    test_acc = (y_pred == y_true).mean()
    print(f"Test accuracy: {test_acc:.3f}")

    # 7) Confusion matrix
    cm = confusion_matrix(y_pred, y_true, n_classes)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Label order:", names)

if __name__ == "__main__":
    main()
