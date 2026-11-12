'use strict';
// Minimal in-memory stand-in for the Fabric stub, so contract logic can be
// tested without a network. It mimics composite keys, range queries, events,
// timestamps and caller identity.
const Module = require('module');
const origLoad = Module._load;
Module._load = function (request, ...rest) {
    if (request === 'fabric-contract-api') {
        return { Contract: class { constructor(name) { this.name = name; } } };
    }
    return origLoad.call(this, request, ...rest);
};

const SEP = '\u0000';
class Ledger {
    constructor() { this.state = new Map(); this.events = []; this.clock = 1794300000; this.txn = 0; }
    ctx(mspId) {
        const ledger = this; const txId = `tx${++this.txn}`; const seconds = ++this.clock;
        const stub = {
            createCompositeKey: (type, attrs) => SEP + type + SEP + attrs.map((a) => a + SEP).join(''),
            splitCompositeKey: (key) => { const p = key.split(SEP); return { objectType: p[1], attributes: p.slice(2, -1) }; },
            getState: async (k) => ledger.state.get(k) || Buffer.alloc(0),
            putState: async (k, v) => { ledger.state.set(k, Buffer.from(v)); },
            deleteState: async (k) => { ledger.state.delete(k); },
            getStateByPartialCompositeKey: (type, attrs) => {
                const prefix = SEP + type + SEP + attrs.map((a) => a + SEP).join('');
                const rows = [...ledger.state.entries()].filter(([k]) => k.startsWith(prefix)).sort(([a], [b]) => (a < b ? -1 : 1));
                return (async function* () { for (const [key, value] of rows) { yield { key, value }; } })();
            },
            getTxTimestamp: () => ({ seconds: { low: seconds, toNumber: () => seconds }, nanos: 0 }),
            getTxID: () => txId,
            setEvent: (name, payload) => { ledger.events.push({ name, payload: JSON.parse(payload.toString()) }); },
        };
        return { stub, clientIdentity: { getMSPID: () => mspId } };
    }
}
module.exports = { Ledger };
