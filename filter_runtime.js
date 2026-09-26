// Direct correlation with the exact effective kernel and reflected source halo.
class RuntimeFilter {
    constructor(config) {
        this.config = config; this.completed = 0;
        this.output = new Float64Array(config.width * config.height * 3);
    }
    calculate(pixel) {
        const c = this.config, x = pixel % c.width, y = Math.floor(pixel / c.width);
        const sums = [0, 0, 0];
        for (let ky = 0; ky < c.size; ky++) for (let kx = 0; kx < c.size; kx++) {
            const weight = c.kernel[ky * c.size + kx], offset = ((y + ky) * c.patchWidth + x + kx) * 3;
            for (let channel = 0; channel < 3; channel++) sums[channel] += c.pixels[offset + channel] * weight;
        }
        for (let channel = 0; channel < 3; channel++) this.output[pixel * 3 + channel] = sums[channel];
    }
    seek(count, budgetMs = Infinity) {
        count = Math.max(0, Math.min(this.config.width * this.config.height, count));
        if (count < this.completed) this.output.fill(0, count * 3);
        if (count < this.completed) this.completed = count;
        const started = performance.now();
        while (this.completed < count) {
            this.calculate(this.completed++);
            if (performance.now() - started >= budgetMs) break;
        }
    }
    terms(pixel, channel) {
        const c = this.config, x = pixel % c.width, y = Math.floor(pixel / c.width), result = [];
        for (let ky = 0; ky < c.size; ky++) for (let kx = 0; kx < c.size; kx++) {
            const value = c.pixels[((y + ky) * c.patchWidth + x + kx) * 3 + channel], weight = c.kernel[ky * c.size + kx];
            result.push({ x: kx - Math.floor(c.size / 2), y: ky - Math.floor(c.size / 2), value, weight, product: value * weight });
        }
        return result;
    }
}
