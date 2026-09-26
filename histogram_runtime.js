// Count source pixels only as their rows are processed.
class RuntimeHistogram {
    constructor(config) {
        this.width = config.width; this.height = config.height;
        this.completed = 0; this.combined = config.combined;
        this.sourceNames = config.sourceNames;
        this.pixels = {}; this.counts = {};
        for (const [name, encoded] of Object.entries(config.pixels)) {
            const bytes = atob(encoded);
            const values = new Uint16Array(bytes.length / 2);
            for (let i = 0; i < values.length; i++) values[i] = bytes.charCodeAt(i * 2) | (bytes.charCodeAt(i * 2 + 1) << 8);
            this.pixels[name] = values; this.counts[name] = new Float64Array(256);
        }
        if (this.combined) this.counts.Total = new Float64Array(256);
    }
    addRow(row, sign) {
        const start = row * this.width, end = start + this.width;
        for (const [name, pixels] of Object.entries(this.pixels)) {
            for (let i = start; i < end; i++) {
                const bin = pixels[i];
                if (bin > 255) continue;
                this.counts[name][bin] += sign;
                if (this.combined && this.sourceNames.includes(name)) this.counts.Total[bin] += sign;
            }
        }
    }
    seek(rows) {
        rows = Math.max(0, Math.min(this.height, rows));
        while (this.completed < rows) this.addRow(this.completed++, 1);
        while (this.completed > rows) this.addRow(--this.completed, -1);
    }
    paint() {
        let maximum = 1;
        for (const counts of Object.values(this.counts)) for (const value of counts) maximum = Math.max(maximum, value);
        maximum *= 1.08;
        document.querySelectorAll('[data-count-tick]').forEach(tick => {
            tick.textContent = Math.round(Number(tick.dataset.countTick) * maximum).toLocaleString();
        });
        for (const [name, counts] of Object.entries(this.counts)) {
            const points = Array.from(counts, (value, i) => (58 + i / 255 * 548).toFixed(2) + ',' + (244 - value / maximum * 202).toFixed(2));
            const curve = 'M ' + points.join(' L ');
            document.getElementById('curve-' + name).setAttribute('d', curve);
            document.getElementById('fill-' + name).setAttribute('d', curve + ' L 606,244 L 58,244 Z');
        }
    }
}
