"""Browser-rendered histogram curves with a replayable drawing animation."""
from html import escape

import streamlit as st

COLORS = {"R": "#db5555", "G": "#23916c", "B": "#487bd5",
          "Y": "#9b770d", "Cb": "#7854b7", "Cr": "#cc547b",
          "Total": "#243b34", "Luminance": "#7a8580"}


def animated_histogram(channels, title, *, combined=False, luminance=False):
    import base64
    import numpy as np
    sources = dict(channels)
    if luminance:
        sources["Luminance"] = .299 * channels["R"] + .587 * channels["G"] + .114 * channels["B"]
    pixels = {}
    for name, channel in sources.items():
        valid = np.isfinite(channel) & (channel >= 0) & (channel <= 256)
        bins = np.full(channel.shape, 256, dtype="<u2")
        bins[valid] = np.minimum(np.floor(channel[valid]), 255).astype("<u2")
        pixels[name] = base64.b64encode(bins.tobytes()).decode("ascii")
    names = list(sources) + (["Total"] if combined else [])
    paths, legend, grid = [], [], []
    for name in names:
        color = COLORS.get(name, "#243b34")
        dash = ' stroke-dasharray="7 4"' if name in ("Total", "Luminance") else ""
        paths.append(f'<path id="fill-{escape(name)}" fill="{color}" opacity=".10"/>'
                     f'<path id="curve-{escape(name)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        legend.append(f'<span><i style="background:{color}"></i>{escape(name)}</span>')
    for fraction in (0, .25, .5, .75, 1):
        y = 244 - fraction * 202
        grid.append(f'<line x1="58" x2="606" y1="{y}" y2="{y}" stroke="#e7ece9"/>'
                    f'<text x="50" y="{y+4}" text-anchor="end" data-count-tick="{fraction}">0</text>')
    for value in (0, 64, 128, 192, 255):
        x = 58 + value / 255 * 548
        grid.append(f'<text x="{x}" y="265" text-anchor="middle">{value}</text>')
    chart = ('<svg viewBox="0 0 640 300" role="img" aria-label="' + escape(title, quote=True) + '">'
             + "".join(grid) + "".join(paths) +
             '<text x="332" y="291" text-anchor="middle">Pixel intensity</text>'
             '<text transform="translate(13 143) rotate(-90)" text-anchor="middle">Count</text></svg>'
             '<div class="legend">' + "".join(legend) + '</div>')
    h, w = next(iter(channels.values())).shape
    runtime = dict(kind="histogram", pixels=pixels, sourceNames=list(channels), combined=combined, width=w, height=h)
    render_player(title, chart, 548, h, "source rows", runtime=runtime)


def prepare_spectrum_channels(channels, center=False):
    import cv2
    import numpy as np
    prepared = {}
    for name, channel in channels.items():
        h, w = channel.shape
        scale = min(1, 128 / max(h, w))
        preview = cv2.resize(channel, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        prepared[name] = preview - preview.mean() if center else preview
    bound = max(1.0, max(float(np.abs(channel).sum()) for channel in prepared.values()))
    return prepared, bound


def animated_spectrum(channel, title, *, bound, log_scale=True):
    import numpy as np
    import matplotlib.pyplot as plt
    h, w = channel.shape
    palette = (plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
    runtime = dict(pixels=channel.ravel().tolist(), width=w, height=h, palette=palette,
                   maximum=float(np.log1p(bound) if log_scale else bound), logScale=log_scale)
    chart = f'<canvas id="spectrum" width="{w}" height="{h}" aria-label="{escape(title, quote=True)}"></canvas>'
    render_player(title, chart, w, h, "source rows", runtime=runtime)


def animated_reconstruction(spectra, title):
    h, w = next(iter(spectra.values())).shape
    values = {name: dict(real=spectrum.real.ravel().tolist(), imag=spectrum.imag.ravel().tolist())
              for name, spectrum in spectra.items()}
    runtime = dict(kind="reconstruction", width=w, height=h, spectra=values)
    chart = f'<canvas id="reconstruction" width="{w}" height="{h}" aria-label="{escape(title, quote=True)}"></canvas>'
    render_player(title, chart, w, h, "frequency rows", runtime=runtime)


def render_player(title, chart, extent, steps, unit, runtime=None):
    import json
    from pathlib import Path
    config = json.dumps(dict(extent=extent, steps=steps, unit=unit, duration=8, runtime=runtime))
    histogram = runtime and runtime.get("kind") == "histogram"
    reconstruction = runtime and runtime.get("kind") == "reconstruction"
    engine_name = "reconstruction_runtime.js" if reconstruction else ("histogram_runtime.js" if histogram else "spectrum_runtime.js")
    engine = Path(__file__).with_name(engine_name).read_text(encoding="utf-8")
    note = ("Live row-by-row Fourier calculation on a preview up to 128 pixels per side. "
            "Scrubbing adds or subtracts row contributions; time is paced playback." if runtime else
            "Progressive reveal of the computed result; time shows animation playback.")
    if histogram:
        note = "Live pixel counting, row by row at full image resolution. Scrub to add or remove rows. Count axis auto-scales; time is paced playback."
    if reconstruction:
        note = "Live inverse Fourier calculation from frequency rows, low vertical frequencies first. Preview up to 128 pixels per side. Time is paced playback."
    st.iframe("""<!doctype html><html><head><meta charset="utf-8"><style>
    body { margin:0; color:#243b34; background:white; font:13px system-ui,sans-serif; }
    .card { border:1px solid #e2e9e5; border-radius:12px; padding:12px; }
    header { font-weight:600; margin-bottom:8px; }
    button, select { cursor:pointer; background:#edf5f0; color:#243b34;
        border:1px solid #d4e4da; border-radius:6px; padding:5px 8px; font:inherit; }
    canvas { display:block; width:100%; height:250px; object-fit:contain; image-rendering:pixelated; }
    svg { display:block; width:100%; height:250px; } text { fill:#63736e; font:11px system-ui; }
    .legend, .controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:8px 0; }
    i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
    input { width:100%; accent-color:#23916c; margin:8px 0; }
    .time { font-variant-numeric:tabular-nums; } .note { font-size:11px; color:#63736e; }
    </style></head><body><div class="card"><header>""" + escape(title) + "</header>" + chart + """
    <label for="timeline">Animation timeline</label>
    <input id="timeline" aria-label="Animation timeline" type="range" min="0" max="8" step="0.01" value="0">
    <div class="time" id="time" aria-live="off"></div>
    <div class="controls"><button id="play">Pause</button><button id="replay">Replay</button>
    <label>Speed <select id="speed" aria-label="Playback speed">
    <option value="0.5">0.5x</option><option value="1" selected>1x</option>
    <option value="2">2x</option><option value="4">4x</option></select></label></div>
    <div class="note">""" + note + """</div>
    </div><script>""" + engine + """
    const config = """ + config + """;
    const histogram = config.runtime.kind === "histogram";
    const reconstruction = config.runtime.kind === "reconstruction";
    const runtime = reconstruction ? new RuntimeReconstruction(config.runtime) : histogram ? new RuntimeHistogram(config.runtime) : new RuntimeSpectrum(config.runtime.pixels, config.runtime.width, config.runtime.height);
    const timeline = document.getElementById('timeline');
    const play = document.getElementById('play');
    const speed = document.getElementById('speed');
    const reveal = document.getElementById('reveal-window');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let position = reduced ? config.duration : 0;
    let playing = !reduced, previous = null, frame = null;
    function draw() {
        const count = Math.min(config.steps, Math.floor(position / config.duration * config.steps));
        if (runtime) {
            runtime.seek(count);
            if (reconstruction) runtime.paint(document.getElementById('reconstruction'));
            else if (histogram) runtime.paint();
            else runtime.paint(document.getElementById('spectrum'), config.runtime.palette, config.runtime.maximum, config.runtime.logScale);
        } else reveal.setAttribute('width', config.extent * count / config.steps);
        timeline.value = position;
        document.getElementById('time').textContent = position.toFixed(1) + ' / ' + config.duration.toFixed(1)
            + ' s | ' + count + ' / ' + config.steps + ' ' + config.unit;
        play.textContent = playing ? 'Pause' : 'Play';
    }
    function tick(now) {
        frame = null;
        if (!playing) return;
        if (previous !== null) position = Math.min(config.duration, position + (now - previous) / 1000 * Number(speed.value));
        previous = now;
        if (position >= config.duration) playing = false;
        draw();
        if (playing) frame = requestAnimationFrame(tick);
    }
    function start() {
        if (frame !== null) cancelAnimationFrame(frame);
        previous = null; playing = true; draw(); frame = requestAnimationFrame(tick);
    }
    play.addEventListener('click', () => {
        if (playing) { playing = false; draw(); }
        else { if (position >= config.duration) position = 0; start(); }
    });
    document.getElementById('replay').addEventListener('click', () => { position = 0; start(); });
    timeline.addEventListener('input', () => { playing = false; position = Number(timeline.value); previous = null; draw(); });
    speed.addEventListener('change', () => { previous = null; });
    draw(); if (playing) start();
    </script></body></html>""", height=460)
