'use strict';
// Caliper workload: registers submissions with random fingerprints. With probability duplicateRate
// a registration is a near-copy (few flipped bits) of an earlier one, to exercise the alert path.
const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const crypto = require('crypto');
const BITS = 256;

class Register extends WorkloadModuleBase {
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
        this.recent = [];
    }
    flip(hex, n) {
        const b = Buffer.from(hex, 'hex');
        for (let i = 0; i < n; i++) { const k = crypto.randomInt(BITS); b[k >> 3] ^= 1 << (k & 7); }
        return b.toString('hex');
    }
    async submitTransaction() {
        let fpH = crypto.randomBytes(BITS / 8).toString('hex');
        let fpB = crypto.randomBytes(BITS / 8).toString('hex');
        if (this.recent.length && Math.random() < (this.roundArguments.duplicateRate || 0)) {
            const r = this.recent[crypto.randomInt(this.recent.length)];
            fpH = this.flip(r.fpH, 12); fpB = this.flip(r.fpB, 12);
        }
        this.recent.push({ fpH, fpB }); if (this.recent.length > 200) { this.recent.shift(); }
        const author = crypto.createHash('sha256').update(crypto.randomBytes(8)).digest('hex');
        await this.sutAdapter.sendRequests({
            contractId: 'submitchain', contractFunction: 'RegisterSubmission',
            contractArguments: [JSON.stringify([author]), fpH, fpB, crypto.randomBytes(16).toString('hex')],
            readOnly: false,
        });
    }
}
module.exports.createWorkloadModule = () => new Register();
