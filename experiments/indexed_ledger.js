'use strict';
// State store for benchmarking: like test/mock.js, but partial-composite-key queries are answered
// from a prefix index instead of a scan of the whole store, as LevelDB/CouchDB would answer them.
// Without this, the benchmark would measure the mock, not the contract.
require('../chaincode/test/mock'); // installs the fabric-contract-api stub
const SEP = '\u0000';

class IndexedLedger {
    constructor() { this.state = new Map(); this.buckets = new Map(); this.events = []; this.clock = 1794300000; this.txn = 0; }
    // Only the prefixes the contract queries are indexed (memory): every attribute-free type prefix,
    // except for the LSH entries, which are queried by (view, table, key).
    _prefixes(key) {
        const p = key.split(SEP);
        if (p[1] === 'lsh') { return [SEP + p.slice(1, 5).join(SEP) + SEP]; }
        return [SEP + p[1] + SEP];
    }
    _put(k, v) {
        if (!this.state.has(k) && k.startsWith(SEP)) {
            for (const pre of this._prefixes(k)) { (this.buckets.get(pre) || this.buckets.set(pre, new Set()).get(pre)).add(k); }
        }
        this.state.set(k, Buffer.from(v));
    }
    _del(k) {
        if (this.state.delete(k) && k.startsWith(SEP)) { for (const pre of this._prefixes(k)) { this.buckets.get(pre).delete(k); } }
    }
    ctx(mspId) {
        const L = this; const txId = `tx${++this.txn}`; const seconds = ++this.clock;
        const stub = {
            createCompositeKey: (type, attrs) => SEP + type + SEP + attrs.map((a) => a + SEP).join(''),
            splitCompositeKey: (key) => { const p = key.split(SEP); return { objectType: p[1], attributes: p.slice(2, -1) }; },
            getState: async (k) => L.state.get(k) || Buffer.alloc(0),
            putState: async (k, v) => { L._put(k, v); },
            deleteState: async (k) => { L._del(k); },
            getStateByPartialCompositeKey: (type, attrs) => {
                const prefix = SEP + type + SEP + attrs.map((a) => a + SEP).join('');
                const keys = [...(L.buckets.get(prefix) || [])].sort();
                return (async function* () { for (const key of keys) { yield { key, value: L.state.get(key) }; } })();
            },
            getTxTimestamp: () => ({ seconds: { low: seconds, toNumber: () => seconds }, nanos: 0 }),
            getTxID: () => txId,
            setEvent: (name, payload) => { L.events.push({ name, payload: JSON.parse(payload.toString()) }); },
        };
        return { stub, clientIdentity: { getMSPID: () => mspId } };
    }
}
module.exports = { IndexedLedger };
