const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const template = fs.readFileSync(path.join(root, 'filter_player.html'), 'utf8');
const engine = fs.readFileSync(path.join(root, 'filter_runtime.js'), 'utf8');
const config = {width: 3, height: 1, patchWidth: 3, size: 1, originX: 0, originY: 0,
    pixels: [10,20,30,40,50,60,70,80,90], kernel: [2]};
const elements = {};
const context = {createImageData: (w,h) => ({data: new Uint8ClampedArray(w*h*4)}),
    putImageData() {}, strokeRect() {}};
const sandbox = {document: {getElementById(id) {
    return elements[id] ??= {value: id === 'speed' ? '1' : '0', open: false,
        getContext: () => context};
}}, matchMedia: () => ({matches: false}), performance: {now: () => 0},
    requestAnimationFrame: () => 1};
vm.createContext(sandbox);
const script = template.split('<script>')[1].split('</script>')[0]
    .replace('/*ENGINE*/', engine).replace('/*CONFIG*/', JSON.stringify(config));
vm.runInContext(script, sandbox);
const pixels = () => Array.from(vm.runInContext('outputImage.data', sandbox));
const original = [10,20,30,255,40,50,60,255,70,80,90,255];
assert.equal((template.match(/<canvas /g) || []).length, 1);
assert.deepEqual(pixels(), original, 'Initial frame must show the original');
vm.runInContext('seek(2)', sandbox);
assert.deepEqual(pixels(), [20,40,60,255,80,100,120,255,70,80,90,255],
    'Only completed pixels should be filtered');
vm.runInContext('seek(1)', sandbox);
assert.deepEqual(pixels(), [20,40,60,255,40,50,60,255,70,80,90,255],
    'Scrubbing backward must restore source pixels');
vm.runInContext('seek(total)', sandbox);
assert.deepEqual(pixels(), [20,40,60,255,80,100,120,255,140,160,180,255]);
assert.deepEqual(Array.from(vm.runInContext('config.pixels', sandbox)), config.pixels,
    'Filtering must never overwrite its source');
elements.replay.onclick();
assert.deepEqual(pixels(), original, 'Replay must restore the complete source');
console.log('Live composite: initial frame, partial progress, rewind, completion, immutable source, and replay passed.');
