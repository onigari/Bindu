"""Generate every figure for the EYE-MAZE deck by running the project's own imgutil code."""
import json, sys, time
sys.path.insert(0, '.')
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from imgutil import color_spaces, compression, wavelet, restoration, filtering
from imgutil.frequency import compute_fft, compute_magnitude_spectrum, reconstruct_partial_image

IN = '/mnt/user-data/uploads/CCC/inputs/'
FONT_DIR = 'fonts/'
from matplotlib import font_manager
for f in ['inter-400-normal.ttf', 'inter-600-normal.ttf']:
    font_manager.fontManager.addfont(FONT_DIR + f)
fam = font_manager.FontProperties(fname=FONT_DIR + 'inter-400-normal.ttf').get_name()
plt.rcParams.update({'font.family': fam, 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.transparent': True})

THEMES = {
    'light': dict(ink='#3B2A20', muted='#8C7565', grid='#E9DCCB', accent='#C8734F',
                  R='#D9776B', G='#7FA87A', B='#6F93BF', Y='#C8734F', extra='#B08968',
                  cmap=LinearSegmentedColormap.from_list('warm', ['#FBF6EF', '#F4C7A6', '#C8734F', '#5A3322'])),
    'dark': dict(ink='#E8E8E0', muted='#8E9388', grid='#2C2E2A', accent='#D4F07A',
                 R='#FF8A80', G='#A5E88F', B='#86B8FF', Y='#D4F07A', extra='#C9B8FF',
                 cmap=LinearSegmentedColormap.from_list('lime', ['#141414', '#2F4A1E', '#8CB84A', '#EAFBB5'])),
}
stats = {}

def load(name, size=None):
    bgr = cv2.imread(IN + name, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return rgb

def save_img(arr, name, q=92):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        cv2.imwrite(f'figs/img/{name}.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        cv2.imwrite(f'figs/img/{name}.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])

def style(ax, t):
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(t['muted'])
    ax.tick_params(colors=t['muted'], labelsize=7.5)
    ax.xaxis.label.set_color(t['ink']); ax.yaxis.label.set_color(t['ink'])
    ax.grid(True, color=t['grid'], lw=0.6); ax.set_axisbelow(True)

# ---------------------------------------------------------------- images
lake = load('white_scratches_lakeside.png')          # 1254x1254, has white scratches
lake512 = cv2.resize(lake, (512, 512), interpolation=cv2.INTER_AREA)
swirl = load('colorful-background-3.png', (480, 270))

# Restoration first so later analysis uses the clean lake image
t0 = time.time()
por = load('white_scratches_portrait.png')
por_fixed, mask = restoration.restore_photo(por, scratches=True, scratch_threshold=30, scratch_size=11)
stats['restore_mask_pct'] = float(np.mean(mask > 0) * 100)
s = 520
dens = cv2.boxFilter((mask > 0).astype(np.float32), -1, (s, s), normalize=True)
yy, xx = np.unravel_index(np.argmax(dens[s//2:-s//2, s//2:-s//2]), dens[s//2:-s//2, s//2:-s//2].shape)
y0, x0 = yy, xx
stats['restore_crop'] = [int(y0), int(x0), s]
save_img(por[y0:y0+s, x0:x0+s], 'restore_before')
save_img(mask[y0:y0+s, x0:x0+s], 'restore_mask')
save_img(por_fixed[y0:y0+s, x0:x0+s], 'restore_after')
save_img(cv2.resize(por, (400, 400), interpolation=cv2.INTER_AREA), 'portrait_full')
save_img(cv2.resize(por_fixed, (400, 400), interpolation=cv2.INTER_AREA), 'portrait_fixed')
repaired, _ = restoration.restore_photo(lake, scratches=True, scratch_threshold=40, scratch_size=7)
gray = cv2.cvtColor(por[y0:y0+s, x0:x0+s], cv2.COLOR_RGB2GRAY)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((11, 11), np.uint8))
save_img(np.clip(tophat * 4, 0, 255), 'restore_tophat')
save_img(cv2.resize(lake, (360, 360), interpolation=cv2.INTER_AREA), 'lake_full')

clean = cv2.resize(repaired, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float64)

# Channels (swirl)
save_img(swirl, 'swirl')
for i, c in enumerate('RGB'):
    tint = np.zeros_like(swirl); tint[:, :, i] = swirl[:, :, i]
    save_img(tint, f'swirl_{c}')
    save_img(swirl[:, :, i], f'swirl_{c}_gray')

# Partial reconstruction (lake)
spectra = {c: compute_fft(clean[:, :, i]) for i, c in enumerate('RGB')}
save_img(clean, 'lake_clean')
for combo in ('RG', 'GB', 'R'):
    rec = reconstruct_partial_image({c: spectra[c] for c in combo})
    save_img(rec, f'recon_{combo}')

# YCbCr
ycc = color_spaces.rgb_to_ycbcr(clean)
for i, c in enumerate(['Y', 'Cb', 'Cr']):
    ch = ycc[:, :, i] if c == 'Y' else 128 + (ycc[:, :, i] - 128) * 5   # chroma stretched x5 for display
    save_img(ch, f'ycc_{c}')
def energy_share(img):
    e = [np.sum((img[:, :, i] - img[:, :, i].mean()) ** 2) for i in range(3)]
    return list(np.array(e) / sum(e) * 100)
stats['share_rgb'] = energy_share(clean)
stats['share_ycc'] = energy_share(ycc)
stats['corr_RG'] = float(np.corrcoef(clean[:, :, 0].ravel(), clean[:, :, 1].ravel())[0, 1])
stats['corr_GB'] = float(np.corrcoef(clean[:, :, 1].ravel(), clean[:, :, 2].ravel())[0, 1])

# Filter examples (crop)
crop = clean[120:376, 60:316]
save_img(crop, 'filt_orig')
save_img(filtering.gaussian_blur(crop, sigma=3), 'filt_gauss')
save_img(filtering.unsharp_mask(crop, sigma=2, amount=1.5), 'filt_unsharp')
save_img(filtering.laplacian_sharpen(crop, ksize=3, scale=0.8), 'filt_lap')

# Lossy Fourier pruning, with the project's own codec
u8 = np.rint(clean).astype(np.uint8)
raw = u8.size
levels = [80, 90, 95, 98, 99, 99.5, 99.8, 99.9, 99.95]
rows = []
for e in levels:
    data, st = compression.compress_image(u8.astype(np.float64), e)
    dec = compression.decompress_image(data)
    mse = float(np.mean((dec.astype(float) - u8) ** 2))
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    kept = sum(s['Kept FFT coefficients'] for s in st) / sum(s['Total FFT coefficients'] for s in st) * 100
    rows.append(dict(energy=e, psnr=psnr, mse=mse, kept=kept, ratio=raw / len(data)))
    if e in (90, 99, 99.9):
        save_img(dec[120:376, 60:316], f'lossy_{str(e).replace(".", "_")}')
stats['lossy'] = rows

# Energy ranking curve (G channel) for the theory plot
F = np.fft.fft2(u8[:, :, 1].astype(float), norm='ortho')
en = np.sort(np.abs(F.ravel()) ** 2)[::-1]
cum = np.cumsum(en) / en.sum() * 100

# Haar subbands + lossless codec
g8 = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
coef = wavelet.haar_transform(np.repeat(g8[:, :, None].astype(np.int64), 1, axis=2), levels=2)[:, :, 0]
vis = np.abs(coef).astype(float)
h2, w2 = 128, 128
disp = np.clip(vis * 4, 0, 255)                     # boost details
disp[:h2, :w2] = vis[:h2, :w2] / max(vis[:h2, :w2].max(), 1) * 255   # LL2 shown as image
save_img(disp, 'haar_subbands')

crop256 = u8[128:384, 128:384].copy()
t1 = time.time()
arch = wavelet.compress_wavelet(crop256, level=9, levels=4)
t_enc = time.time() - t1
back = wavelet.decompress_wavelet(arch)
stats['iwv'] = dict(raw=crop256.size, archive=len(arch), exact=bool(np.array_equal(back, crop256)),
                    ratio=crop256.size / len(arch), enc_s=t_enc)
save_img(crop256, 'iwv_crop')

# Entropy before/after transform (reversible color transform + Haar)
px = crop256.astype(np.int64)
rct = px.copy(); rct[:, :, 0] -= rct[:, :, 1]; rct[:, :, 2] -= rct[:, :, 1]
hc = wavelet.haar_transform(rct, 4)
def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
stats['entropy_pixels'] = H(px.ravel()); stats['entropy_coeffs'] = H(hc.ravel())
stats['zero_frac'] = float(np.mean(np.abs(hc) <= 2) * 100)

# Luminance for histogram
lum = 0.299 * clean[:, :, 0] + 0.587 * clean[:, :, 1] + 0.114 * clean[:, :, 2]

# ---------------------------------------------------------------- plots per theme
for name, t in THEMES.items():
    out = f'figs/{name}/'
    # Histograms
    fig, ax = plt.subplots(figsize=(4.2, 2.25))
    x = np.arange(256)
    for i, c in enumerate('RGB'):
        h, _ = np.histogram(clean[:, :, i], 256, (0, 256))
        ax.fill_between(x, h, color=t[c], alpha=0.28, lw=0); ax.plot(x, h, color=t[c], lw=1.1, label=c)
    h, _ = np.histogram(lum, 256, (0, 256))
    ax.plot(x, h, color=t['ink'], lw=1.2, ls='--', label='Y (luminance)')
    ax.set_xlim(0, 255); ax.set_xlabel('intensity level $k$'); ax.set_ylabel('count $h[k]$')
    ax.set_yticks([]); style(ax, t)
    leg = ax.legend(frameon=False, fontsize=7.5, ncol=4, loc='upper left', labelcolor=t['ink'])
    fig.tight_layout(); fig.savefig(out + 'hist.pdf'); plt.close(fig)

    # Spectra
    for i, c in enumerate('RGB'):
        mag = compute_magnitude_spectrum(compute_fft(clean[:, :, i] - clean[:, :, i].mean()))
        fig = plt.figure(figsize=(2, 2)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mag, cmap=t['cmap'], vmin=np.percentile(mag, 5), vmax=np.percentile(mag, 99.9)); ax.axis('off')
        fig.savefig(out + f'spec_{c}.png', dpi=160); plt.close(fig)

    # Energy share bars
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    xs = np.arange(3); wbar = 0.38
    ax.bar(xs - wbar/2, stats['share_rgb'], wbar, color=[t['R'], t['G'], t['B']], label='R, G, B')
    ax.bar(xs + wbar/2, stats['share_ycc'], wbar, color=t['Y'], alpha=0.9, label='Y, Cb, Cr', hatch='')
    for xi, v in zip(xs + wbar/2, stats['share_ycc']):
        ax.text(xi, v + 2, f'{v:.0f}%', ha='center', fontsize=7, color=t['ink'])
    for xi, v in zip(xs - wbar/2, stats['share_rgb']):
        ax.text(xi, v + 2, f'{v:.0f}%', ha='center', fontsize=7, color=t['muted'])
    ax.set_xticks(xs, ['R | Y', 'G | Cb', 'B | Cr']); ax.set_ylim(0, 128)
    ax.set_ylabel('share of energy (%)'); style(ax, t); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=7, labelcolor=t['ink'], loc='upper right', ncol=2)
    fig.tight_layout(); fig.savefig(out + 'energy_share.pdf'); plt.close(fig)

    # Filter frequency responses (1D slice, normalised frequency)
    N = 512; f = np.fft.rfftfreq(N)
    def resp(k1d):
        k = np.zeros(N); k[:len(k1d)] = k1d; return np.abs(np.fft.rfft(k))
    g = cv2.getGaussianKernel(0 * 0 + 25, 2.0).ravel()
    box = np.ones(5) / 5
    lap = np.array([0, 1, 0]) * 0 + np.array([-1, 3, -1])        # 1D slice of I - Laplacian
    unsharp = -1.0 * g.copy(); unsharp[len(g)//2] += 2.0
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    ax.plot(f, resp(g), color=t['B'], lw=1.6, label='Gaussian blur ($\\sigma$=2)')
    ax.plot(f, resp(box), color=t['G'], lw=1.3, label='box blur (5)')
    ax.plot(f, resp(unsharp), color=t['R'], lw=1.6, label='unsharp mask')
    ax.plot(f, resp(lap), color=t['accent'], lw=1.3, ls='--', label='Laplacian sharpen')
    ax.axhline(1, color=t['muted'], lw=0.7, ls=':')
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 5.3)
    ax.set_xlabel('spatial frequency (cycles / pixel)'); ax.set_ylabel('$|H(\\omega)|$')
    style(ax, t); ax.legend(frameon=False, fontsize=6.5, labelcolor=t['ink'], loc='upper left', bbox_to_anchor=(0, 0.93), ncol=1)
    fig.tight_layout(); fig.savefig(out + 'freq_resp.pdf'); plt.close(fig)

    # Energy compaction curve
    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    frac = np.arange(1, len(cum) + 1) / len(cum) * 100
    ax.plot(frac, cum, color=t['accent'], lw=1.8)
    k99 = frac[np.searchsorted(cum, 99)]
    ax.axhline(99, color=t['muted'], lw=0.8, ls='--'); ax.axvline(k99, color=t['muted'], lw=0.8, ls='--')
    ax.text(0.0013, 63, f'99% energy\nin {k99:.1f}% of\ncoefficients', fontsize=7, color=t['ink'])
    ax.set_xscale('log'); ax.set_xlim(0.001, 100); ax.set_ylim(60, 101)
    ax.set_xlabel('coefficients kept (%, largest first)'); ax.set_ylabel('energy kept (%)')
    style(ax, t); fig.tight_layout(); fig.savefig(out + 'energy_curve.pdf'); plt.close(fig)
    stats['k99_green'] = float(k99)

    # PSNR vs kept coefficients (measured with the project codec)
    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    ks = [r['kept'] for r in rows]; ps = [r['psnr'] for r in rows]
    ax.plot(ks, ps, color=t['accent'], lw=1.6, marker='o', ms=3.5, mfc=t['accent'])
    for r in rows:
        if r['energy'] in (90, 99, 99.9):
            ax.annotate(f"{r['energy']:g}%", (r['kept'], r['psnr']), textcoords='offset points',
                        xytext=(-4, 6), ha='right', fontsize=7, color=t['ink'])
    ax.set_xscale('log'); ax.set_xlabel('FFT coefficients kept (%)'); ax.set_ylabel('PSNR (dB)')
    style(ax, t); fig.tight_layout(); fig.savefig(out + 'psnr_curve.pdf'); plt.close(fig)

    # Value distribution: pixels vs Haar coefficients
    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    v1, c1 = np.unique(px.ravel(), return_counts=True)
    v2, c2 = np.unique(np.clip(hc.ravel(), -128, 255), return_counts=True)
    ax.plot(v1, c1 / c1.sum() * 100, color=t['B'], lw=1.4, label='raw pixels')
    ax.plot(v2, c2 / c2.sum() * 100, color=t['accent'], lw=1.5, label='Haar coefficients')
    ax.set_xlim(-60, 255); ax.set_xlabel('value'); ax.set_ylabel('share (%)')
    style(ax, t); ax.legend(frameon=False, fontsize=7, labelcolor=t['ink'])
    fig.tight_layout(); fig.savefig(out + 'coef_dist.pdf'); plt.close(fig)

json.dump(stats, open('stats.json', 'w'), indent=1, default=float)
print(json.dumps(stats, indent=1, default=float))
