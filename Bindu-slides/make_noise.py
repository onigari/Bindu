"""Denoising figures, using the project's own imgutil.noise functions."""
import sys, json; sys.path.insert(0, '.')
import numpy as np, cv2
from imgutil import noise
bgr = cv2.imread('/mnt/user-data/uploads/CCC/inputs/white_scratches_lakeside.png')
from imgutil import restoration
lake = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
lake, _ = restoration.restore_photo(lake, scratches=True, scratch_threshold=40, scratch_size=7)
img = cv2.resize(lake, (512, 512), interpolation=cv2.INTER_AREA)[140:396, 200:456]  # boat crop 256x256
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2); return 10 * np.log10(255 ** 2 / mse)
def save(a, n): cv2.imwrite(f'figs/img/{n}.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
out = {}
save(img, 'dn_clean')
for kind, amt, tag in [('Gaussian', 25.0, 'g'), ('Salt and pepper', 10.0, 'sp')]:
    noisy = noise.add_noise(img, kind, amt, seed=42); save(noisy, f'dn_{tag}_noisy')
    out[f'{tag}_noisy'] = psnr(noisy, img)
    for m, mt in [('Median', 'med'), ('Gaussian blur', 'gb'), ('Non-local means', 'nlm')]:
        r = noise.remove_noise(noisy, m, size=5 if m != 'Non-local means' else 3, strength=15.0)
        save(r, f'dn_{tag}_{mt}'); out[f'{tag}_{mt}'] = psnr(r, img)
print(json.dumps({k: round(v, 1) for k, v in out.items()}, indent=0))
json.dump(out, open('noise_stats.json', 'w'))
