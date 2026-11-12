'use strict';
// Microbenchmark of the SubmitChain contract logic against the in-memory state store used by the
// unit tests. Measures (a) state reads per registration, a platform-independent cost, and (b) wall
// time of the contract code. It does NOT measure Fabric endorsement, ordering or commit latency.
const crypto = require('crypto');
const { IndexedLedger: Ledger } = require('./indexed_ledger');
const Contract = require('../chaincode/lib/submitchain');

const BITS = 256;
const ACTIVE = (process.argv[2] || '100,1000,5000,20000').split(',').map(Number);
const PROBES = Number(process.argv[3] || 200);
const TABLES = Number(process.env.TABLES || 64), TBITS = Number(process.env.TBITS || 16);
const MODES = (process.env.MODES || 'scan,index').split(',');
const fp = () => crypto.randomBytes(BITS / 8).toString('hex');
const author = () => JSON.stringify([crypto.createHash('sha256').update(crypto.randomBytes(8)).digest('hex')]);

function instrument(ledger) {
    const counters = { keysRead: 0, rangeQueries: 0 };
    const orig = ledger.ctx.bind(ledger);
    ledger.ctx = (msp) => {
        const c = orig(msp); const s = c.stub;
        const g = s.getState; s.getState = async (k) => { counters.keysRead++; return g(k); };
        const r = s.getStateByPartialCompositeKey;
        s.getStateByPartialCompositeKey = (t, a) => {
            counters.rangeQueries++; const it = r(t, a);
            return (async function* () { for await (const x of it) { counters.keysRead++; yield x; } })();
        };
        return c;
    };
    return counters;
}

(async () => {
    const rows = [];
    for (const mode of MODES) {
        for (const n of ACTIVE) {
            const ledger = new Ledger(); const c = new Contract();
            // Preload in index mode (cheap); state layout is identical in both modes, then switch mode.
            await c.SetConfig(ledger.ctx('Org1'), JSON.stringify({ bits: BITS, tauH: 40, tauB: 40, mode: 'index', tables: TABLES, tableBits: TBITS, seed: 'bench' }));
            const orgs = ['Org1', 'Org2', 'Org3'];
            for (let i = 0; i < n; i++) {
                await c.RegisterSubmission(ledger.ctx(orgs[i % 3]), author(), fp(), fp(), crypto.randomBytes(8).toString('hex'));
            }
            await c.SetConfig(ledger.ctx('Org1'), JSON.stringify({ bits: BITS, tauH: 40, tauB: 40, mode, tables: TABLES, tableBits: TBITS, seed: 'bench' }));
            let bytes = 0; for (const [k, v] of ledger.state) { bytes += Buffer.byteLength(k) + v.length; }
            const perSub = bytes / n;
            const counters = instrument(ledger);
            const times = [];
            for (let i = 0; i < PROBES; i++) {
                const ctx = ledger.ctx(orgs[i % 3]);
                const before = counters.keysRead;
                const t0 = process.hrtime.bigint();
                const r = JSON.parse(await c.RegisterSubmission(ctx, author(), fp(), fp(), crypto.randomBytes(8).toString('hex')));
                times.push({ ms: Number(process.hrtime.bigint() - t0) / 1e6, reads: counters.keysRead - before });
                await c.UpdateStatus(ledger.ctx(orgs[i % 3]), r.sid, 'WITHDRAWN'); // keep the active set at n
            }
            times.sort((a, b) => a.ms - b.ms);
            const reads = times.map((t) => t.reads).sort((a, b) => a - b);
            const row = { mode, active: n, median_ms: times[Math.floor(PROBES / 2)].ms, p95_ms: times[Math.floor(PROBES * 0.95)].ms,
                median_reads: reads[Math.floor(PROBES / 2)], state_bytes_per_submission: Math.round(perSub) };
            rows.push(row); console.error(JSON.stringify(row));
        }
    }
    console.log(JSON.stringify({ node: process.version, bits: BITS, index: { tables: TABLES, tableBits: TBITS }, probes: PROBES, rows }, null, 2));
})();
