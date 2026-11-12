'use strict';
const { Contract } = require('fabric-contract-api');
const L = require('./logic');

const CONFIG_KEY = 'config';
const SUB = 'sub';        // sub~<sid>                       -> submission record
const ACTIVE = 'active';  // active~<org>~<sid>              -> {fpH, fpB, authors}
const BAND = 'lsh';       // lsh~<view>~<table>~<key>~<org>~<sid>  -> marker
const MARK = Buffer.from('\u0001');

function txTime(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    let s = ts.seconds;
    if (s && typeof s === 'object') { s = typeof s.toNumber === 'function' ? s.toNumber() : s.low; }
    return new Date(Number(s) * 1000).toISOString();
}

async function getJSON(ctx, key) {
    const raw = await ctx.stub.getState(key);
    return raw && raw.length ? JSON.parse(raw.toString()) : null;
}

async function putJSON(ctx, key, obj) {
    await ctx.stub.putState(key, Buffer.from(JSON.stringify(obj)));
}

class SubmitChainContract extends Contract {
    constructor() { super('SubmitChain'); this._posMemo = null; this._positions = null; }

    // ---- configuration -----------------------------------------------------

    /** Set fingerprint length, thresholds and matching mode. Only the organisation
     *  that first set the configuration may change it (prototype governance). */
    async SetConfig(ctx, configJSON) {
        const cfg = L.validateConfig(JSON.parse(configJSON));
        const caller = ctx.clientIdentity.getMSPID();
        const old = await getJSON(ctx, CONFIG_KEY);
        if (old && old.admin !== caller) { throw new Error('only the configuring organisation may change the configuration'); }
        if (old && ['bits', 'tables', 'tableBits', 'seed'].some((k) => old[k] !== cfg[k])) {
            throw new Error('bits and index parameters cannot change once set: existing entries would become incomparable');
        }
        cfg.admin = caller;
        await putJSON(ctx, CONFIG_KEY, cfg);
        return JSON.stringify(cfg);
    }

    async GetConfig(ctx) {
        const cfg = await getJSON(ctx, CONFIG_KEY);
        if (!cfg) { throw new Error('configuration not set: call SetConfig first'); }
        return JSON.stringify(cfg);
    }

    async _config(ctx) {
        const cfg = JSON.parse(await this.GetConfig(ctx));
        const memo = `${cfg.bits}|${cfg.tables}|${cfg.tableBits}|${cfg.seed}`;
        if (this._posMemo !== memo) { this._positions = L.tablePositions(cfg.bits, cfg.tables, cfg.tableBits, cfg.seed); this._posMemo = memo; }
        cfg.positions = this._positions;
        return cfg;
    }

    // ---- registration and matching ----------------------------------------

    /**
     * Register a submission and screen it against active submissions of other organisations.
     * @param authorsJSON JSON array of author pseudonyms (HMAC-SHA256 hex); first = corresponding author
     * @param fpH header-view fingerprint (hex)   @param fpB body-view fingerprint (hex)
     * @param nonce random string from the gateway (makes the identifier unpredictable)
     */
    async RegisterSubmission(ctx, authorsJSON, fpH, fpB, nonce) {
        const cfg = await this._config(ctx);
        const authors = JSON.parse(authorsJSON);
        if (!Array.isArray(authors) || authors.length === 0 || !authors.every(L.isPseudonym)) {
            throw new Error('authors must be a non-empty array of 64-character hex pseudonyms');
        }
        if (!L.isFingerprint(fpH, cfg.bits) || !L.isFingerprint(fpB, cfg.bits)) {
            throw new Error(`fingerprints must be ${cfg.bits / 4} lowercase hex characters`);
        }
        if (typeof nonce !== 'string' || nonce.length < 8) { throw new Error('nonce must be at least 8 characters'); }

        const org = ctx.clientIdentity.getMSPID();
        const now = txTime(ctx);
        const sid = L.submissionId(authors[0], org, now, nonce);
        const subKey = ctx.stub.createCompositeKey(SUB, [sid]);
        if (await getJSON(ctx, subKey)) { throw new Error('submission identifier already registered'); }

        const candidates = cfg.mode === 'index'
            ? await this._candidatesByIndex(ctx, cfg, org, fpH, fpB)
            : await this._candidatesByScan(ctx, org);

        const matches = [];
        for (const c of candidates) {
            const dH = L.hamming(fpH, c.fpH);
            const dB = L.hamming(fpB, c.fpB);
            if (L.isMatch(dH, dB, cfg.tauH, cfg.tauB)) {
                const shared = c.authors.some((a) => authors.includes(a));
                matches.push({ sid: c.sid, org: c.org, dH, dB, sharedAuthor: shared });
            }
        }
        matches.sort((x, y) => (x.sid < y.sid ? -1 : 1)); // deterministic order

        const record = {
            sid, org, authors, fpH, fpB,
            status: L.STATUS.UNDER_REVIEW,
            history: [{ status: L.STATUS.UNDER_REVIEW, time: now, txId: ctx.stub.getTxID() }],
            alerts: matches.map((m) => ({ ...m, time: now })),
        };
        await putJSON(ctx, subKey, record);
        await putJSON(ctx, ctx.stub.createCompositeKey(ACTIVE, [org, sid]), { fpH, fpB, authors });
        for (const [view, fp] of [['h', fpH], ['b', fpB]]) {
            for (const [idx, val] of L.tableKeys(fp, cfg.positions)) {
                await ctx.stub.putState(ctx.stub.createCompositeKey(BAND, [view, idx, val, org, sid]), MARK);
            }
        }
        for (const m of matches) {
            const otherKey = ctx.stub.createCompositeKey(SUB, [m.sid]);
            const other = await getJSON(ctx, otherKey);
            other.alerts.push({ sid, org, dH: m.dH, dB: m.dB, sharedAuthor: m.sharedAuthor, time: now });
            await putJSON(ctx, otherKey, other);
        }
        if (matches.length) {
            // Fabric keeps one event per transaction, so all matches travel in one payload.
            ctx.stub.setEvent('DuplicateAlert', Buffer.from(JSON.stringify({ sid, org, matches })));
        }
        return JSON.stringify({ sid, registeredAt: now, matches });
    }

    async _candidatesByScan(ctx, org) {
        const out = [];
        for await (const r of ctx.stub.getStateByPartialCompositeKey(ACTIVE, [])) {
            const { attributes } = ctx.stub.splitCompositeKey(r.key);
            if (attributes[0] === org) { continue; }
            const v = JSON.parse(r.value.toString());
            out.push({ org: attributes[0], sid: attributes[1], fpH: v.fpH, fpB: v.fpB, authors: v.authors });
        }
        return out;
    }

    async _candidatesByIndex(ctx, cfg, org, fpH, fpB) {
        const seen = new Map();
        for (const [view, fp] of [['h', fpH], ['b', fpB]]) {
            for (const [idx, val] of L.tableKeys(fp, cfg.positions)) {
                for await (const r of ctx.stub.getStateByPartialCompositeKey(BAND, [view, idx, val])) {
                    const { attributes } = ctx.stub.splitCompositeKey(r.key);
                    const cOrg = attributes[3]; const cSid = attributes[4];
                    if (cOrg !== org && !seen.has(cSid)) { seen.set(cSid, cOrg); }
                }
            }
        }
        const out = [];
        for (const cSid of [...seen.keys()].sort()) {
            const cOrg = seen.get(cSid);
            const v = await getJSON(ctx, ctx.stub.createCompositeKey(ACTIVE, [cOrg, cSid]));
            if (v) { out.push({ org: cOrg, sid: cSid, fpH: v.fpH, fpB: v.fpB, authors: v.authors }); }
        }
        return out;
    }

    // ---- lifecycle ---------------------------------------------------------

    async UpdateStatus(ctx, sid, status) {
        if (!L.TERMINAL.has(status)) { throw new Error('status must be REJECTED, WITHDRAWN or ACCEPTED'); }
        const cfg = await this._config(ctx);
        const key = ctx.stub.createCompositeKey(SUB, [sid]);
        const rec = await getJSON(ctx, key);
        if (!rec) { throw new Error('no such submission'); }
        const org = ctx.clientIdentity.getMSPID();
        if (rec.org !== org) { throw new Error('only the registering organisation may change the status'); }
        if (rec.status !== L.STATUS.UNDER_REVIEW) { throw new Error('submission already has a final status'); }
        rec.status = status;
        rec.history.push({ status, time: txTime(ctx), txId: ctx.stub.getTxID() });
        await putJSON(ctx, key, rec);
        await ctx.stub.deleteState(ctx.stub.createCompositeKey(ACTIVE, [org, sid]));
        for (const [view, fp] of [['h', rec.fpH], ['b', rec.fpB]]) {
            for (const [idx, val] of L.tableKeys(fp, cfg.positions)) {
                await ctx.stub.deleteState(ctx.stub.createCompositeKey(BAND, [view, idx, val, org, sid]));
            }
        }
        return JSON.stringify({ sid, status });
    }

    // ---- explorer queries (read-only) --------------------------------------

    /** Author view. The gateway authenticates the author's ORCID iD and passes the derived pseudonym. */
    async GetReceipt(ctx, sid, authorPseudonym) {
        const rec = await getJSON(ctx, ctx.stub.createCompositeKey(SUB, [sid]));
        if (!rec) { throw new Error('no such submission'); }
        if (rec.org !== ctx.clientIdentity.getMSPID()) { throw new Error('receipts are issued by the registering organisation'); }
        if (!rec.authors.includes(authorPseudonym)) { throw new Error('caller is not an author of this submission'); }
        return JSON.stringify({ sid: rec.sid, org: rec.org, status: rec.status, history: rec.history });
    }

    /** Member-editor view: alerts for a submission the caller's organisation owns. */
    async CheckActiveMatch(ctx, sid) {
        const rec = await getJSON(ctx, ctx.stub.createCompositeKey(SUB, [sid]));
        if (!rec) { throw new Error('no such submission'); }
        if (rec.org !== ctx.clientIdentity.getMSPID()) { throw new Error('only the registering organisation may query matches'); }
        const alerts = [];
        for (const a of rec.alerts) {
            const other = await getJSON(ctx, ctx.stub.createCompositeKey(SUB, [a.sid]));
            alerts.push({ ...a, otherActive: other.status === L.STATUS.UNDER_REVIEW });
        }
        return JSON.stringify({ sid, status: rec.status, activeMatch: alerts.some((a) => a.otherActive), alerts });
    }

    /** Public view: existence and registration time only. */
    async VerifyExists(ctx, sid) {
        const rec = await getJSON(ctx, ctx.stub.createCompositeKey(SUB, [sid]));
        return JSON.stringify(rec ? { registered: true, registeredAt: rec.history[0].time } : { registered: false });
    }
}

module.exports = SubmitChainContract;
