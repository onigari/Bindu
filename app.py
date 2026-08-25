import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Color Channel Analyzer", layout="wide")
st.title("Color Channel Analyzer & Frequency Pruning")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Processing Controls")
channel_selection = st.sidebar.selectbox("Select Channel to Analyze", ["Red", "Green", "Blue"])
prune_percentile = st.sidebar.slider("Harmonic Pruning (Discard Bottom X%)", 0.0, 99.9, 0.0, 0.1)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Load Image and Convert to NumPy Array
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    # 2. Separate Channels
    # Indexing: 0 = Red, 1 = Green, 2 = Blue
    channel_idx = {"Red": 0, "Green": 1, "Blue": 2}[channel_selection]
    selected_channel = img_array[:, :, channel_idx]

    # --- MATH & PROCESSING ---
    # 3. Compute 2D DFT
    dft = fft2(selected_channel)
    dft_shifted = fftshift(dft)
    magnitude_spectrum = np.abs(dft_shifted)
    
    # 4. Harmonic Energy Pruning
    # Calculate the threshold value based on the selected percentile
    threshold_val = np.percentile(magnitude_spectrum, prune_percentile)
    
    # Create a mask and apply it to the shifted DFT
    dft_pruned = np.where(magnitude_spectrum >= threshold_val, dft_shifted, 0)
    
    # 5. Inverse Transform for Reconstruction
    dft_ishift = ifftshift(dft_pruned)
    reconstructed_channel = np.abs(ifft2(dft_ishift))

    # --- VISUALIZATION ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Channel")
        st.image(selected_channel, cmap="gray", clamp=True)
        
    with col2:
        st.subheader("Frequency Spectrum")
        # Log scale used for visual clarity of the frequency spectrum
        log_spectrum = np.log1p(np.abs(dft_pruned))
        st.image(log_spectrum / np.max(log_spectrum), clamp=True)
        
    with col3:
        st.subheader("Reconstructed Channel")
        st.image(reconstructed_channel, cmap="gray", clamp=True)
        
    # --- HISTOGRAM ---
    st.subheader(f"Intensity Histogram: {channel_selection} Channel")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(reconstructed_channel.ravel(), bins=256, range=(0, 256), color=channel_selection.lower())
    st.pyplot(fig)

else:
    st.info("Please upload an image to begin the analysis.")