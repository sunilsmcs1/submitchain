'use strict';
// Empirical recall of the contract's LSH index (single view): fraction of pairs at Hamming distance d
// that share at least one table key, using the contract's own tablePositions/tableKeys.
const crypto = require('crypto');
const L = require('../chaincode/lib/logic');
const BITS = 256; const TRIALS = 2000;
function flip(fp, n) {
    const b = Buffer.from(fp, 'hex'); const used = new Set();
    while (used.size < n) { used.add(crypto.randomInt(BITS)); }
    for (const k of used) { b[k >> 3] ^= 1 << (7 - (k & 7)); }
    return b.toString('hex');
}
const out = {};
for (const [tables, k] of [[64, 16], [128, 20]]) {
    const pos = L.tablePositions(BITS, tables, k, 'recall');
    const row = [];
    for (let d = 0; d <= 80; d += 8) {
        let hit = 0;
        for (let t = 0; t < TRIALS; t++) {
            const a = crypto.randomBytes(BITS / 8).toString('hex'); const b = flip(a, d);
            const ka = L.tableKeys(a, pos); const kb = L.tableKeys(b, pos);
            if (ka.some(([i, v], j) => kb[j][1] === v)) { hit++; }
        }
        row.push([d, hit / TRIALS]);
    }
    out[`L${tables}_k${k}`] = row;
}
console.log(JSON.stringify({ trials: TRIALS, bits: BITS, recall: out }));
