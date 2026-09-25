# Bindu

### White Scratch Removal

Choose **White Scratch Removal** in the sidebar after uploading a photo or
selecting a sample. This tool repairs white dots and bright scratches only.

- **Scratch threshold:** defaults to 30. Increase it to select fewer marks;
  lower it to detect fainter damage.
- **Scratch detection width:** defaults to 11 pixels. Choose a window wider
  than the dots or scratch thickness.

Click **Repair white dots / scratches** to process the image. Changing the
image or settings requires another click. Inspect the repair mask, compare
results with the wipe at the bottom, and download the repaired PNG.

Detection uses a grayscale white top-hat operation, thresholding, and a
one-pixel mask expansion. Telea inpainting fills selected pixels from their
surroundings. Masks covering over 10% of the image are rejected. The tool may
mistake genuine bright details for damage; it cannot recover missing original
detail with certainty. Images are limited to 4 million pixels.

Uploads remain selected when switching between the other image tools.

TODO:
- [x] separate an input image into red, green, and blue channels
- [x] display and compare the histograms of the separated channels
- [x] compare the frequency content of the separated channels
- [x] apply filters (smoothing/sharpening effects) individually on each channel
- [x] enable partial image reconstruction using the frequency content of 1-2 channels
- [x] compare color spaces (e.g. RGB vs YCbCr)
- [x] compress images via harmonic energy pruning
- [x] reconstruct a compressed/pruned image and compare it against the original image
- [x] lossless integer wavelet compression and standalone decompression

Run `streamlit run app.py`, upload an image or choose a sample, then select
**Partial image reconstruction** in the sidebar. Choose R, G, B, R+G, R+B,
or G+B to preview and download the reconstructed PNG. Selected channels are
reconstructed from their full complex FFTs; omitted channels are set to zero.

Select **Color space comparison** to compare RGB and full-range YCbCr channels
as grayscale previews, histograms with shared axes, and frequency spectra with
a shared color scale. Y represents luma; Cb and Cr represent color differences
with neutral chroma at 128. Mean removal is enabled by default for the spectra
to highlight spatial detail rather than constant channel offsets. This tool
does not perform compression or chroma subsampling.

Select **Compression and decompression** and choose **Compress**. Set the energy
retention target per RGB channel and click **Compress and preview**. The tool
ranks Fourier conjugate pairs by energy, retains the strongest pairs (always
keeping the mean/DC component), and saves their indices and complex coefficients
in a compressed `.npz` archive. Previews compare the original against the actual
decoded archive, with MSE, PSNR, an amplified difference image, and file sizes.
Input is rounded to 8-bit RGB; coefficients are stored as complex64.

Download the compressed archive, then choose **Decompress** and upload it to
restore the image and download a PNG. No original image is needed for decoding.
Pruning is lossy: decompression cannot restore discarded detail. At 100% energy,
all Fourier pairs are retained, subject to floating-point storage precision.
Energy retention is not a compression ratio; archives may exceed PNG/JPEG or
even raw RGB size. The displayed ratio compares raw 8-bit RGB bytes with archive
bytes, and a separately encoded original PNG size is also shown. Images are
limited to 4 million pixels.

### Lossless wavelet compression and decompression

Select **Lossless wavelet compression** in the sidebar:

1. Choose **Compress**, upload an image or select a sample, and set wavelet
   decomposition levels (1–8) and compression effort (0–9). Every setting
   preserves the same pixels. More levels analyze larger spatial regions.
2. Click **Compress and verify**. The tool applies reversible integer Haar
   wavelets and DEFLATE, decodes the result, and checks exact pixel equality.
   It shows both images, raw RGB size, archive size, and their size ratio.
3. Download the custom **.iwv** archive. This is not a PNG or JPEG 2000 file;
   ordinary image viewers cannot open it directly.
4. Choose **Decompress** and upload that archive. No original image is required.
   Preview the decoded pixels and download an uncompressed BMP.

Lossless means exact preservation of the **8-bit RGB pixels supplied to the
encoder**, not the original uploaded file bytes. The existing image loader
converts uploads to RGB (discarding alpha), and floating-point sample values
are rounded before encoding. Metadata and original color profiles are not
preserved. Decompression also accepts legacy non-interlaced, single-frame
8-bit RGB PNGs without transparency. The limits are 4 million pixels and
64 MB per archive upload.

The wavelet transform retains every coefficient without quantization. File
sizes depend on image content: small or noisy images may grow, and archives
can be larger than PNG or JPEG. The displayed ratio compares archive bytes
with raw RGB bytes, not the original
uploaded file size. An uncompressed BMP also includes a header and row padding.

### Custom codec implementation

The lossless tool implements the algorithms directly, without Pillow, OpenCV
encoding/decoding, or a compression-library call in its wavelet codec:

- `imgutil/wavelet.py`: reversible color differences (R-G, G, B-G), multilevel
  integer Haar lifting, signed zigzag packing, byte-plane ordering, archive
  framing, inverse transforms, and decoded-pixel CRC verification.
  For each pair `a,b`, lifting computes `d=b-a` and `s=a+floor(d/2)`.
  The inverse computes `a=s-floor(d/2)` and `b=a+d`, exactly even for negative
  details. Rows then columns form 2D low/detail bands; the next level transforms
  only the low/low region. Odd tails are carried unchanged. Decoding reverses
  level order and then axis order.
- IWV version 1 stores a 17-byte big-endian header (`IWV1`, width uint32,
  height uint32, levels uint8, RGB CRC-32 uint32), followed by a zlib stream.
  The stream holds four byte planes of little-endian uint32 zigzag coefficients,
  flattened in row/column/channel order. No coefficients are discarded.

- `imgutil/lossless.py`: PNG chunk parsing/writing, CRC-32 checksums, all five
  row filters (None, Sub, Up, Average, Paeth), adaptive filter selection,
  reversing filters, and uncompressed 24-bit BMP export.
- `imgutil/deflate.py`: bit packing, a 32 KiB LZ77 sliding window, canonical
  fixed Huffman encoding, stored-block fallback, zlib framing, and Adler-32.
  The decoder also reads dynamic Huffman tables and multiple DEFLATE blocks.
- Level 0 writes stored DEFLATE blocks. Levels 1–9 increase the number of
  previous matches searched; all preserve identical pixels. This encoder
  uses fixed Huffman codes rather than generating dynamic tables. Higher
  levels do not guarantee a smaller file for every input.

This educational Python implementation is slower and may produce larger files
than optimized codecs. NumPy handles arrays; it does not perform compression.
General image imports (such as JPEG) use the existing OpenCV dependency. All
PNG downloads use the custom encoder. The application has no direct Pillow
calls, although third-party UI/plotting packages may depend on Pillow internally.
The separate Fourier `.npz` archive tool continues to use NumPy's ZIP storage.

Run `python -m unittest discover -s tests -v` to check known Haar coefficients,
exact round trips for noise, constant images, checkerboards, odd dimensions,
single-pixel axes, multiple decomposition/effort settings, and damaged archives.

Format references: [PNG specification](https://www.w3.org/TR/png/) and
[DEFLATE RFC 1951](https://www.rfc-editor.org/rfc/rfc1951).

### Noise lab

Choose **Noise lab** to add Gaussian or salt-and-pepper noise, remove existing
noise, or add and remove noise in one run. A fixed random seed makes the added
noise repeatable. Removal offers median filtering, Gaussian blur, and non-local
means. Adjust the filter width or denoising strength, then press the action
button to process. Input or setting changes require another click.

The shared sidebar image stays unchanged. Download noisy or denoised PNGs and
inspect the comparisons at the bottom. In combined mode the removal wipe
compares the noisy input against the denoised result. Denoising can soften
image detail and does not guarantee recovery of the original pixels.
