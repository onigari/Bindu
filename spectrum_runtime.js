// Incremental separable DFT: each source row contributes to every frequency bin.
class RuntimeSpectrum {
    constructor(pixels, width, height) {
        this.pixels = pixels; this.width = width; this.height = height;
        this.real = new Float64Array(width * height);
        this.imag = new Float64Array(width * height);
        this.rows = []; this.completed = 0;
        this.cosX = new Float64Array(width * width);
        this.sinX = new Float64Array(width * width);
        for (let k = 0; k < width; k++) for (let x = 0; x < width; x++) {
            const angle = -2 * Math.PI * k * x / width;
            this.cosX[k * width + x] = Math.cos(angle);
            this.sinX[k * width + x] = Math.sin(angle);
        }
    }
    addRow(y, sign) {
        const w = this.width, h = this.height;
        if (!this.rows[y]) {
            const re = new Float64Array(w), im = new Float64Array(w);
            for (let k = 0; k < w; k++) for (let x = 0; x < w; x++) {
                const value = this.pixels[y * w + x];
                re[k] += value * this.cosX[k * w + x];
                im[k] += value * this.sinX[k * w + x];
            }
            this.rows[y] = [re, im];
        }
        const [re, im] = this.rows[y];
        for (let ky = 0; ky < h; ky++) {
            const angle = -2 * Math.PI * ky * y / h;
            const c = Math.cos(angle), s = Math.sin(angle);
            for (let kx = 0; kx < w; kx++) {
                const i = ky * w + kx;
                this.real[i] += sign * (re[kx] * c - im[kx] * s);
                this.imag[i] += sign * (re[kx] * s + im[kx] * c);
            }
        }
    }
    seek(rows) {
        rows = Math.max(0, Math.min(this.height, rows));
        while (this.completed < rows) this.addRow(this.completed++, 1);
        while (this.completed > rows) this.addRow(--this.completed, -1);
        if (!rows) { this.real.fill(0); this.imag.fill(0); }
    }
    paint(canvas, palette, maximum, logScale) {
        const w = this.width, h = this.height, ctx = canvas.getContext('2d');
        const image = ctx.createImageData(w, h);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const src = ((y + Math.ceil(h / 2)) % h) * w + (x + Math.ceil(w / 2)) % w;
            let magnitude = Math.hypot(this.real[src], this.imag[src]);
            if (logScale) magnitude = Math.log1p(magnitude);
            const color = palette[Math.min(255, Math.max(0, Math.round(magnitude / maximum * 255)))];
            const i = (y * w + x) * 4;
            image.data[i] = color[0]; image.data[i+1] = color[1]; image.data[i+2] = color[2]; image.data[i+3] = 255;
        }
        ctx.putImageData(image, 0, 0);
    }
}
