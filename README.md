# Color Channel Analyzer and Image Compressor

TODO:
- [x] separate an input image into red, green, and blue channels
- [x] display and compare the histograms of the separated channels
- [x] compare the frequency content of the separated channels
- [x] apply filters (smoothing/sharpening effects) individually on each channel
- [x] enable partial image reconstruction using the frequency content of 1-2 channels
- [x] compare color spaces (e.g. RGB vs YCbCr)
- [x] compress images via harmonic energy pruning
- [x] reconstruct a compressed/pruned image and compare it against the original image
- [x] lossless PNG compression and standalone decompression

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

### Lossless PNG compression and decompression

Select **Lossless PNG compression** in the sidebar:

1. Choose **Compress**, upload an image or select a sample, and set the PNG
   compression level (0–9). Higher levels spend more effort compressing; every
   level preserves the same pixels.
2. Click **Compress and verify**. The tool encodes PNG using reversible row
   filters and DEFLATE, decodes the result, and checks exact pixel equality.
   It shows both images, raw RGB size, PNG size, and the raw-to-PNG size ratio.
3. Download the lossless PNG.
4. Choose **Decompress** and upload that PNG. No original image is required.
   Preview the decoded pixels and download an uncompressed BMP.

Lossless means exact preservation of the **8-bit RGB pixels supplied to the
encoder**, not the original uploaded file bytes. The existing image loader
converts uploads to RGB (discarding alpha), and floating-point sample values
are rounded before encoding. Metadata and original color profiles are not
preserved. Decoding accepts non-interlaced, single-frame 8-bit RGB PNGs without
transparency; other PNG modes are rejected rather than silently converted.
The limits are 4 million pixels and 64 MB per compressed PNG upload.

Unlike FFT pruning, PNG discards no pixel information. File sizes depend on
image content: small or noisy images may grow, and PNG can be larger than JPEG.
The displayed ratio compares PNG bytes with raw RGB bytes, not the original
uploaded file size. An uncompressed BMP also includes a header and row padding.

### Custom codec implementation

The lossless tool implements the algorithms directly, without Pillow, OpenCV
encoding/decoding, or a compression-library call in its PNG codec:

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

Run `python -m unittest discover -s tests -v` to check pixel preservation,
all PNG filters, checksum failures, BMP layout, and compatibility with stored,
fixed, and dynamic DEFLATE streams. Standard-library `zlib` is used only as an
independent reference in tests, never by the custom codec.

Format references: [PNG specification](https://www.w3.org/TR/png/) and
[DEFLATE RFC 1951](https://www.rfc-editor.org/rfc/rfc1951).
