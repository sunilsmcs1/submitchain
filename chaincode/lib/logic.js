'use strict';
// Pure, deterministic helpers. Integer operations only (no floating point),
// so every endorsing peer computes identical results.
const crypto = require('crypto');

const POP = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
const STATUS = Object.freeze({
    UNDER_REVIEW: 'UNDER_REVIEW',
    REJECTED: 'REJECTED',
    WITHDRAWN: 'WITHDRAWN',
    ACCEPTED: 'ACCEPTED',
});
const TERMINAL = new Set([STATUS.REJECTED, STATUS.WITHDRAWN, STATUS.ACCEPTED]);

function sha256Hex(s) {
    return crypto.createHash('sha256').update(s, 'utf8').digest('hex');
}

/** True if fp is a lowercase hex string encoding exactly `bits` bits. */
function isFingerprint(fp, bits) {
    return typeof fp === 'string' && fp.length === bits / 4 && /^[0-9a-f]+$/.test(fp);
}

function isPseudonym(a) {
    return typeof a === 'string' && /^[0-9a-f]{64}$/.test(a);
}

/** Hamming distance between two equal-length hex strings. */
function hamming(a, b) {
    if (a.length !== b.length) {
        throw new Error('fingerprints differ in length');
    }
    let d = 0;
    for (let i = 0; i < a.length; i++) {
        d += POP[parseInt(a[i], 16) ^ parseInt(b[i], 16)];
    }
    return d;
}

/** Bit i of a hex fingerprint (bit 0 = most significant bit of the first hex digit). */
function bitAt(fp, i) {
    return (parseInt(fp[i >> 2], 16) >> (3 - (i & 3))) & 1;
}

/**
 * Deterministic bit-sampling tables for the LSH index: table t uses `tableBits` distinct positions
 * drawn from a SHA-256-seeded Fisher-Yates shuffle of 0..bits-1. Same seed -> same tables on every peer.
 */
function tablePositions(bits, tables, tableBits, seed) {
    const out = [];
    let counter = 0; let pool = Buffer.alloc(0);
    const nextU32 = () => {
        if (pool.length < 4) { pool = crypto.createHash('sha256').update(`${seed}|${counter++}`).digest(); }
        const v = pool.readUInt32BE(0); pool = pool.subarray(4); return v;
    };
    for (let t = 0; t < tables; t++) {
        const perm = Array.from({ length: bits }, (_, i) => i);
        for (let i = bits - 1; i > 0; i--) {
            const j = nextU32() % (i + 1); const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
        out.push(perm.slice(0, tableBits).sort((a, b) => a - b));
    }
    return out;
}

/** Index keys of a fingerprint: [[tableIndex, keyHex], ...]. */
function tableKeys(fp, positions) {
    return positions.map((pos, t) => {
        let hex = ''; let nib = 0;
        pos.forEach((p, i) => {
            nib = (nib << 1) | bitAt(fp, p);
            if ((i & 3) === 3) { hex += nib.toString(16); nib = 0; }
        });
        if (pos.length & 3) { hex += (nib << (4 - (pos.length & 3))).toString(16); }
        return [String(t), hex];
    });
}

/** Decision rule of Section 2.4: flag when either view is within its threshold. */
function isMatch(dH, dB, tauH, tauB) {
    return dH <= tauH || dB <= tauB;
}

function submissionId(corrAuthor, org, isoTime, nonce) {
    return sha256Hex(`${corrAuthor}|${org}|${isoTime}|${nonce}`);
}

function validateConfig(c) {
    if (![64, 128, 256, 512].includes(c.bits)) { throw new Error('config.bits must be 64, 128, 256 or 512'); }
    for (const k of ['tauH', 'tauB']) {
        if (!Number.isInteger(c[k]) || c[k] < 0 || c[k] > c.bits) {
            throw new Error(`config.${k} must be an integer between 0 and bits`);
        }
    }
    if (!['scan', 'index'].includes(c.mode)) { throw new Error('config.mode must be "scan" or "index"'); }
    if (!Number.isInteger(c.tables) || c.tables < 1 || c.tables > 256) { throw new Error('config.tables must be an integer from 1 to 256'); }
    if (!Number.isInteger(c.tableBits) || c.tableBits < 4 || c.tableBits > 32 || c.tableBits > c.bits) {
        throw new Error('config.tableBits must be an integer from 4 to 32');
    }
    if (typeof c.seed !== 'string' || c.seed.length < 1) { throw new Error('config.seed must be a non-empty string'); }
    return { bits: c.bits, tauH: c.tauH, tauB: c.tauB, mode: c.mode, tables: c.tables, tableBits: c.tableBits, seed: c.seed };
}

module.exports = {
    STATUS, TERMINAL, sha256Hex, isFingerprint, isPseudonym, hamming, bitAt, tablePositions, tableKeys, isMatch, submissionId, validateConfig,
};
