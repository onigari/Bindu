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
