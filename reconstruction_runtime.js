// Accumulate actual inverse-DFT contributions, low vertical frequencies first.
class RuntimeReconstruction {
    constructor(config) {
        this.width = config.width; this.height = config.height;
        this.spectra = config.spectra; this.completed = 0;
        this.order = Array.from({length:this.height}, (_, i) => i)
            .sort((a,b) => Math.min(a,this.height-a)-Math.min(b,this.height-b) || a-b);
        this.output = {}; this.rows = {};
        for (const name of Object.keys(this.spectra)) {
            this.output[name] = new Float64Array(this.width*this.height);
            this.rows[name] = [];
        }
    }
    addRow(step, sign) {
        const w=this.width, h=this.height, ky=this.order[step];
        for (const [name, spectrum] of Object.entries(this.spectra)) {
            if (!this.rows[name][ky]) {
                const re=new Float64Array(w), im=new Float64Array(w);
                for (let x=0;x<w;x++) for (let kx=0;kx<w;kx++) {
                    const angle=2*Math.PI*kx*x/w, c=Math.cos(angle), s=Math.sin(angle);
                    const i=ky*w+kx;
                    re[x]+=(spectrum.real[i]*c-spectrum.imag[i]*s)/w;
                    im[x]+=(spectrum.real[i]*s+spectrum.imag[i]*c)/w;
                }
                this.rows[name][ky]=[re,im];
            }
            const [re,im]=this.rows[name][ky], output=this.output[name];
            for (let y=0;y<h;y++) {
                const angle=2*Math.PI*ky*y/h, c=Math.cos(angle), s=Math.sin(angle);
                for (let x=0;x<w;x++) output[y*w+x]+=sign*(re[x]*c-im[x]*s)/h;
            }
        }
    }
    seek(rows) {
        rows=Math.max(0,Math.min(this.height,rows));
        while(this.completed<rows) this.addRow(this.completed++,1);
        while(this.completed>rows) this.addRow(--this.completed,-1);
        if(!rows) for(const output of Object.values(this.output)) output.fill(0);
    }
    paint(canvas) {
        const ctx=canvas.getContext('2d'), image=ctx.createImageData(this.width,this.height);
        for(let i=0;i<this.width*this.height;i++) {
            for(const [name,values] of Object.entries(this.output))
                image.data[i*4+'RGB'.indexOf(name)]=Math.round(Math.max(0,Math.min(255,values[i])));
            image.data[i*4+3]=255;
        }
        ctx.putImageData(image,0,0);
    }
}
