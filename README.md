# Color Channel Analyzer and Image Compressor

TODO:
- [x] separate an input image into red, green, and blue channels
- [x] display and compare the histograms of the separated channels
- [x] compare the frequency content of the separated channels
- [x] apply filters (smoothing/sharpening effects) individually on each channel
- [x] enable partial image reconstruction using the frequency content of 1-2 channels
- [ ] compare color spaces (e.g. RGB vs YCbCr)
- [ ] compress images via harmonic energy pruning
- [ ] reconstruct a compressed/pruned image and compare it against the original image

Run `streamlit run app.py`, upload an image or choose a sample, then select
**Partial image reconstruction** in the sidebar. Choose R, G, B, R+G, R+B,
or G+B to preview and download the reconstructed PNG. Selected channels are
reconstructed from their full complex FFTs; omitted channels are set to zero.
