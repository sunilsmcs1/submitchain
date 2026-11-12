'use strict';
// Command-line client a journal gateway uses to talk to its Fabric peer.
// NOT run in the development sandbox (no Fabric network there): test it on your network first.
//
//   node client.js config '{"bits":256,"tauH":44,"tauB":44,"mode":"index","tables":64,"tableBits":16,"seed":"<consortium seed>"}'
//   node client.js register '["<author pseudonym>"]' <fpH hex> <fpB hex>
//   node client.js status <sid> REJECTED|WITHDRAWN|ACCEPTED
//   node client.js receipt <sid> <author pseudonym>
//   node client.js match <sid>
//   node client.js verify <sid>
//
// Connection settings come from environment variables (defaults match fabric-samples test-network, Org1).
const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const env = (k, d) => process.env[k] || d;
const ORG_DIR = env('ORG_DIR', path.resolve(process.env.HOME, 'fabric-samples/test-network/organizations/peerOrganizations/org1.example.com'));
const MSP_ID = env('MSP_ID', 'Org1MSP');
const PEER_ENDPOINT = env('PEER_ENDPOINT', 'localhost:7051');
const PEER_HOST = env('PEER_HOST', 'peer0.org1.example.com');
const CHANNEL = env('CHANNEL', 'mychannel');
const CHAINCODE = env('CHAINCODE', 'submitchain');
const USER = env('FABRIC_USER', 'User1@org1.example.com');

const first = (dir) => path.join(dir, fs.readdirSync(dir)[0]);

async function main() {
    const [cmd, ...args] = process.argv.slice(2);
    const tlsCert = fs.readFileSync(path.join(ORG_DIR, 'peers', PEER_HOST, 'tls', 'ca.crt'));
    const client = new grpc.Client(PEER_ENDPOINT, grpc.credentials.createSsl(tlsCert), { 'grpc.ssl_target_name_override': PEER_HOST });
    const mspDir = path.join(ORG_DIR, 'users', USER, 'msp');
    const gateway = connect({
        client,
        identity: { mspId: MSP_ID, credentials: fs.readFileSync(first(path.join(mspDir, 'signcerts'))) },
        signer: signers.newPrivateKeySigner(crypto.createPrivateKey(fs.readFileSync(first(path.join(mspDir, 'keystore'))))),
    });
    try {
        const contract = gateway.getNetwork(CHANNEL).getContract(CHAINCODE);
        const submit = async (fn, ...a) => Buffer.from(await contract.submitTransaction(fn, ...a)).toString();
        const query = async (fn, ...a) => Buffer.from(await contract.evaluateTransaction(fn, ...a)).toString();
        let out;
        switch (cmd) {
        case 'config': out = await submit('SetConfig', args[0]); break;
        case 'register': out = await submit('RegisterSubmission', args[0], args[1], args[2], crypto.randomBytes(16).toString('hex')); break;
        case 'status': out = await submit('UpdateStatus', args[0], args[1]); break;
        case 'receipt': out = await query('GetReceipt', args[0], args[1]); break;
        case 'match': out = await query('CheckActiveMatch', args[0]); break;
        case 'verify': out = await query('VerifyExists', args[0]); break;
        default: throw new Error('unknown command; see the header of this file');
        }
        console.log(out);
    } finally {
        gateway.close(); client.close();
    }
}
main().catch((e) => { console.error(e.message || e); process.exit(1); });
