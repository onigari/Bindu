from pathlib import Path
p = Path('output/presentation/project_presentation.tex')
s = p.read_text()
header = s.split('% SLIDE 1:')[0]
fourier = s[s.index('% SLIDE 3:'):s.index('% SLIDE 4:')]
fourier = fourier.replace('SLIDE 3: 1:00--1:50 (50 seconds)', 'SLIDE 4: 2:15--3:15 (60 seconds)')
first = r'''% SLIDE 1: 0:00--0:45 (45 seconds)
% Our project lets users explore how images work through an interactive
% Streamlit application. The channel analyzer starts by separating an image
% into red, green and blue components. A grayscale preview shows the intensity
% of each component, while a colored preview shows its contribution to the
% original image. Histograms count pixels at each intensity, helping us compare
% how the channels use their available range. The two-dimensional FFT gives a
% different view: it describes spatial frequencies rather than pixel positions.
% We display log-magnitude spectra so weaker frequency components remain visible.
% Together, these views connect the image's colors with its intensity and
% frequency information before we modify or reconstruct it.
\begin{frame}{Channel analyzer}
\textbf{Inspect each RGB component in the spatial and frequency domains}
\vspace{5mm}
\begin{itemize}\setlength\itemsep{4mm}
\item \textbf{Channel previews:} isolate red, green and blue contributions.
\item \textbf{Histograms:} compare pixel intensity distributions.
\item \textbf{2-D FFT spectra:} inspect spatial frequency content.
\end{itemize}
\vspace{4mm}
\[F_c(u,v)=\mathcal F\{I_c(x,y)\},\qquad c\in\{R,G,B\}\]
\vspace{2mm}
\muted{Log-magnitude display makes weaker frequency components easier to see.}
\end{frame}

% SLIDE 2: 0:45--1:30 (45 seconds)
% Partial reconstruction asks what happens when we keep only some color
% channels. The app computes the full complex FFT of each selected channel,
% then applies the inverse FFT to recover its spatial values. Both magnitude
% and phase matter in this reconstruction. Channels that are not selected
% become zero. For example, choosing red and green preserves those components
% and removes blue, changing the overall color balance. Users can select one
% channel or any pair, compare the result with the original and download it.
% This operation removes entire color channels. Within each selected channel,
% it retains the full frequency spectrum, so it differs from the coefficient
% pruning used later in Fourier compression.
\begin{frame}{Partial image reconstruction}
\begin{columns}[T,onlytextwidth]
\column{0.49\textwidth}
\textbf{Keep one channel or a pair}
\begin{itemize}\setlength\itemsep{4mm}
\item Choose R, G, B, R+G, R+B or G+B
\item Apply the inverse FFT to selected channels
\item Set omitted channels to zero
\end{itemize}
\column{0.46\textwidth}
\textbf{Full complex spectra}
\vspace{4mm}
\[\widehat I_c=\begin{cases}
\mathcal F^{-1}(F_c),&c\text{ selected},\\
0,&c\text{ omitted}.
\end{cases}\]
\vspace{3mm}
\muted{Example: R+G preserves red and green values and removes blue.}
\end{columns}
\vspace{6mm}
\alert{Selected channels retain both magnitude and phase.}
\end{frame}

% SLIDE 3: 1:30--2:15 (45 seconds)
% The color-space comparison shows the same image in RGB and full-range
% YCbCr. RGB expresses each pixel through red, green and blue intensities.
% YCbCr instead separates luma, called Y, from two color-difference channels,
% Cb and Cr. The displayed formula shows that green contributes the most to
% the luma calculation used here. A neutral color has chroma values of 128.
% The app compares channel previews, histograms and frequency spectra. Shared
% axes and a shared spectrum scale support fair visual comparison. By default,
% it removes channel means before plotting spectra so constant offsets do not
% dominate. This feature demonstrates the representation change without
% performing compression or chroma subsampling.
\begin{frame}{Color spaces: RGB and YCbCr}
\begin{columns}[T,onlytextwidth]
\column{0.45\textwidth}
\textbf{RGB}
\par\vspace{3mm} Red, green and blue intensities describe each pixel.
\par\vspace{6mm}
\textbf{YCbCr}
\par\vspace{3mm} Y represents luma. Cb and Cr represent color differences.
\column{0.50\textwidth}
\textbf{Full-range BT.601 conversion}
\vspace{3mm}
\[Y=0.299R+0.587G+0.114B\]
\muted{Neutral chroma: Cb = Cr = 128.}
\par\vspace{5mm} Compare grayscale previews, histograms and FFT spectra.
\end{columns}
\vspace{6mm}
\small\muted{Shared plotting scales support comparison. Mean removal is on by default for spectra. This tool does not compress or subsample the image.}
\end{frame}

'''
last = r'''% SLIDE 5: 3:15--4:00 (45 seconds)
% Finally, the image filters let us modify local image structure. Gaussian
% blur uses weighted neighbors, while box blur uses their average. Both smooth
% the image, with stronger settings also removing fine detail. For sharpening,
% the app provides Laplacian sharpening and unsharp masking. Unsharp masking
% subtracts a blurred image to isolate detail, then adds a scaled version of
% that detail back to the original. Users can compare filtering only red,
% green or blue against filtering all channels. Applying the same filter
% independently to every channel matches its whole-image counterpart within
% floating-point tolerance. These controls show the tradeoff between smoothing
% detail and emphasizing edges. Excessive sharpening can amplify noise or halos.
\begin{frame}{Image filters}
\begin{columns}[T,onlytextwidth]
\column{0.46\textwidth}
\textbf{Smoothing}
\begin{itemize}\setlength\itemsep{3mm}
\item Gaussian blur: weighted neighbors
\item Box blur: local averaging
\end{itemize}
\par\vspace{5mm}\muted{Larger blur settings smooth more detail.}
\column{0.49\textwidth}
\textbf{Sharpening}
\begin{itemize}\setlength\itemsep{3mm}
\item Laplacian: emphasize local changes
\item Unsharp mask: add detail back
\end{itemize}
\[I_{\mathrm{sharp}}=I+a\bigl(I-G_\sigma*I\bigr)\]
\small\muted{$G_\sigma*I$: Gaussian-blurred image.\\$a$: sharpening amount.}
\end{columns}
\vspace{5mm}
\alert{Compare R-only, G-only, B-only and all-channel filtering.}
\par\vspace{3mm}
\small\muted{Strong sharpening can amplify noise and produce halos.}
\end{frame}
\end{document}
'''
p.write_text(header + first + fourier + last)
