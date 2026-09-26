"""Kernel figures for the 'one tool, four kernels' slide.

Uses the project's own imgutil.filtering: filter_kernel() gives the weights,
and the matching whole-image filter gives the output image. Each kernel is
also checked against the filter by convolving it with the crop directly.
Run from the Bindu-slides folder with the imgutil package importable.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from imgutil import filtering

FONT_DIR = 'fonts/'
for f in ['inter-400-normal.ttf', 'inter-600-normal.ttf']:
    font_manager.fontManager.addfont(FONT_DIR + f)
fam = font_manager.FontProperties(fname=FONT_DIR + 'inter-400-normal.ttf').get_name()
plt.rcParams.update({'font.family': fam, 'savefig.transparent': True})

THEMES = {
    'light': dict(ink='#3B2A20', muted='#8C7565', zero='#FBF6EF', pos='#C8734F', neg='#6F93BF', line='#E6D2BC'),
    'dark':  dict(ink='#E8E8E0', muted='#8E9388', zero='#1A1B19', pos='#D4F07A', neg='#86B8FF', line='#2C2E2A'),
}

# Same settings as the images on the Filtering slide
FILTERS = [
    ('box',     'Box blur',          dict(ksize=5),              filtering.box_blur),
    ('gauss',   'Gaussian blur',     dict(sigma=3),              filtering.gaussian_blur),
    ('lap',     'Laplacian sharpen', dict(ksize=3, scale=0.8),   filtering.laplacian_sharpen),
    ('unsharp', 'Unsharp mask',      dict(sigma=2, amount=1.5),  filtering.unsharp_mask),
]

crop = cv2.cvtColor(cv2.imread('figs/img/filt_orig.jpg'), cv2.COLOR_BGR2RGB).astype(np.float64)


def draw_kernel(k, t, path):
    n = k.shape[0]
    # positive and negative weights scaled separately (sqrt), so a small negative
    # ring stays visible next to a large positive centre
    pos, neg = k.max(), -k.min()
    v = np.where(k > 0, np.sqrt(np.clip(k, 0, None) / pos), 0.0)
    if neg > 0:
        v = np.where(k < 0, -np.sqrt(np.clip(-k, 0, None) / neg), v)
    cmap = LinearSegmentedColormap.from_list('div', [t['neg'], t['zero'], t['pos']])
    fig, ax = plt.subplots(figsize=(1.3, 1.3))
    ax.imshow(v, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
    if n <= 7:
        for i in range(n + 1):
            ax.axhline(i - 0.5, color=t['line'], lw=0.6)
            ax.axvline(i - 0.5, color=t['line'], lw=0.6)
        for (i, j), w in np.ndenumerate(k):
            txt = f'{w:.2f}' if n > 3 else (f'{w:.1f}' if w else '0')
            strong = abs(v[i, j]) > 0.75
            ax.text(j, i, txt, ha='center', va='center', fontsize=5.2 if n > 3 else 7,
                    color=t['zero'] if strong else t['ink'])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(t['line']); s.set_linewidth(0.6)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


stats = {}
for key, name, params, fn in FILTERS:
    k = filtering.filter_kernel(name, **params)
    out = fn(crop, **params)
    # the kernel alone should reproduce the tool's output (same default border)
    direct = cv2.filter2D(crop, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT_101)
    stats[key] = dict(size=k.shape[0], sum=float(k.sum()), min=float(k.min()), max=float(k.max()),
                      max_abs_diff=float(np.abs(direct - out).max()))
    out8 = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(f'figs/img/kern_out_{key}.jpg', cv2.cvtColor(out8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    for theme, t in THEMES.items():
        draw_kernel(k, t, f'figs/{theme}/kern_{key}.pdf')

print(json.dumps(stats, indent=1))
