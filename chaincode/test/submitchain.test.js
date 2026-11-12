'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const { Ledger } = require('./mock');
const Contract = require('../lib/submitchain');
const L = require('../lib/logic');

const BITS = 256;
const pseudo = (s) => crypto.createHash('sha256').update(s).digest('hex');
const randFp = () => crypto.randomBytes(BITS / 8).toString('hex');
/** Flip exactly n distinct bits of a hex fingerprint. */
function flip(fp, n) {
    const bytes = Buffer.from(fp, 'hex'); const used = new Set();
    while (used.size < n) { used.add(crypto.randomInt(BITS)); }
    for (const b of used) { bytes[b >> 3] ^= 1 << (b & 7); }
    return bytes.toString('hex');
}
async function setup(mode = 'scan') {
    const ledger = new Ledger(); const c = new Contract();
    await c.SetConfig(ledger.ctx('OrgA'), JSON.stringify({ bits: BITS, tauH: 40, tauB: 40, mode, tables: 64, tableBits: 16, seed: 'test' }));
    return { ledger, c };
}
const reg = async (c, ledger, org, author, fpH, fpB) =>
    JSON.parse(await c.RegisterSubmission(ledger.ctx(org), JSON.stringify([pseudo(author)]), fpH, fpB, crypto.randomBytes(8).toString('hex')));

test('hamming distance, bit extraction and LSH tables', () => {
    assert.equal(L.hamming('ff00', '0f00'), 4);
    assert.equal(L.hamming('abcd', 'abcd'), 0);
    const fp = randFp(); assert.equal(L.hamming(fp, flip(fp, 37)), 37);
    assert.equal(L.bitAt('80', 0), 1); assert.equal(L.bitAt('80', 1), 0); assert.equal(L.bitAt('01', 7), 1);
    const pos = L.tablePositions(BITS, 8, 16, 'seed');
    assert.deepEqual(pos, L.tablePositions(BITS, 8, 16, 'seed'), 'deterministic');
    assert.ok(pos.every((p) => p.length === 16 && new Set(p).size === 16 && p.every((i) => i >= 0 && i < BITS)));
    // contiguous positions reproduce the hex digits
    assert.deepEqual(L.tableKeys('abcdef12', [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), [['0', 'abcd']]);
    assert.deepEqual(L.tableKeys('ff', [[0, 1, 2, 3, 4, 5]]), [['0', 'fc']]); // 6 bits, zero-padded
});

for (const mode of ['scan', 'index']) {
    test(`[${mode}] duplicate at another journal raises an alert on both records`, async () => {
        const { ledger, c } = await setup(mode);
        const h = randFp(); const b = randFp();
        const first = await reg(c, ledger, 'OrgA', 'alice', h, b);
        assert.equal(first.matches.length, 0);
        const second = await reg(c, ledger, 'OrgB', 'alice', flip(h, 20), flip(b, 90));
        assert.equal(second.matches.length, 1);
        assert.equal(second.matches[0].sid, first.sid);
        assert.equal(second.matches[0].dH, 20);
        assert.equal(second.matches[0].sharedAuthor, true);
        assert.equal(ledger.events.at(-1).name, 'DuplicateAlert');
        const seenByA = JSON.parse(await c.CheckActiveMatch(ledger.ctx('OrgA'), first.sid));
        assert.equal(seenByA.activeMatch, true);
        assert.equal(seenByA.alerts[0].org, 'OrgB');
    });

    test(`[${mode}] unrelated papers and same-journal resubmissions do not alert`, async () => {
        const { ledger, c } = await setup(mode);
        const h = randFp(); const b = randFp();
        await reg(c, ledger, 'OrgA', 'alice', h, b);
        assert.equal((await reg(c, ledger, 'OrgB', 'bob', randFp(), randFp())).matches.length, 0);
        assert.equal((await reg(c, ledger, 'OrgA', 'alice', flip(h, 5), flip(b, 5))).matches.length, 0);
    });

    test(`[${mode}] a rejected paper can be submitted elsewhere without an alert`, async () => {
        const { ledger, c } = await setup(mode);
        const h = randFp(); const b = randFp();
        const first = await reg(c, ledger, 'OrgA', 'alice', h, b);
        await c.UpdateStatus(ledger.ctx('OrgA'), first.sid, 'REJECTED');
        const second = await reg(c, ledger, 'OrgB', 'alice', flip(h, 3), flip(b, 3));
        assert.equal(second.matches.length, 0);
        assert.equal([...ledger.state.keys()].filter((k) => k.includes(first.sid) && !k.includes('\u0000sub\u0000')).length, 0,
            'index entries of a closed submission are removed');
    });
}

test('only the owning journal can change status, and terminal states are final', async () => {
    const { ledger, c } = await setup();
    const s = await reg(c, ledger, 'OrgA', 'alice', randFp(), randFp());
    await assert.rejects(c.UpdateStatus(ledger.ctx('OrgB'), s.sid, 'REJECTED'), /registering organisation/);
    await assert.rejects(c.UpdateStatus(ledger.ctx('OrgA'), s.sid, 'UNDER_REVIEW'), /status must be/);
    await c.UpdateStatus(ledger.ctx('OrgA'), s.sid, 'ACCEPTED');
    await assert.rejects(c.UpdateStatus(ledger.ctx('OrgA'), s.sid, 'WITHDRAWN'), /final status/);
});

test('explorer: three disclosure levels', async () => {
    const { ledger, c } = await setup();
    const s = await reg(c, ledger, 'OrgA', 'alice', randFp(), randFp());
    const pub = JSON.parse(await c.VerifyExists(ledger.ctx('OrgC'), s.sid));
    assert.deepEqual(Object.keys(pub).sort(), ['registered', 'registeredAt']);
    assert.equal(JSON.parse(await c.VerifyExists(ledger.ctx('OrgC'), 'f'.repeat(64))).registered, false);
    const receipt = JSON.parse(await c.GetReceipt(ledger.ctx('OrgA'), s.sid, pseudo('alice')));
    assert.equal(receipt.history.length, 1);
    assert.ok(!('fpH' in receipt) && !('authors' in receipt));
    await assert.rejects(c.GetReceipt(ledger.ctx('OrgA'), s.sid, pseudo('mallory')), /not an author/);
    await assert.rejects(c.CheckActiveMatch(ledger.ctx('OrgB'), s.sid), /registering organisation/);
});

test('input validation', async () => {
    const { ledger, c } = await setup();
    const a = JSON.stringify([pseudo('alice')]);
    await assert.rejects(c.RegisterSubmission(ledger.ctx('OrgA'), a, 'abc', randFp(), 'nonce1234'), /fingerprints must be/);
    await assert.rejects(c.RegisterSubmission(ledger.ctx('OrgA'), '["0000-0001-2345-6789"]', randFp(), randFp(), 'nonce1234'), /pseudonyms/);
    const cfg = { bits: BITS, tauH: 1, tauB: 1, mode: 'scan', tables: 64, tableBits: 16, seed: 'test' };
    await assert.rejects(c.SetConfig(ledger.ctx('OrgB'), JSON.stringify(cfg)), /configuring organisation/);
    await assert.rejects(c.SetConfig(ledger.ctx('OrgA'), JSON.stringify({ ...cfg, bits: 128 })), /cannot change/);
    await assert.rejects(c.SetConfig(ledger.ctx('OrgA'), JSON.stringify({ ...cfg, seed: 'other' })), /cannot change/);
    await c.SetConfig(ledger.ctx('OrgA'), JSON.stringify({ ...cfg, tauH: 30, mode: 'index' })); // thresholds and mode may change
});

test('index mode retrieves near-duplicates that scan mode finds (recall on random codes at distances 10-39)', async () => {
    const scan = await setup('scan'); const band = await setup('index');
    let scanHits = 0; let bandHits = 0;
    for (let i = 0; i < 40; i++) {
        const h = randFp(); const b = randFp(); const h2 = flip(h, 10 + (i % 30)); const b2 = flip(b, 10 + (i % 30));
        for (const [env, add] of [[scan, (n) => { scanHits += n; }], [band, (n) => { bandHits += n; }]]) {
            await reg(env.c, env.ledger, 'OrgA', `a${i}`, h, b);
            add((await reg(env.c, env.ledger, 'OrgB', `a${i}`, h2, b2)).matches.filter((m) => m.sharedAuthor).length);
        }
    }
    assert.equal(scanHits, 40);
    assert.ok(bandHits >= 38, `index mode recall too low: ${bandHits}/40`);
});
