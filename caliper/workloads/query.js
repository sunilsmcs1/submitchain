'use strict';
const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const crypto = require('crypto');

class Query extends WorkloadModuleBase {
    async submitTransaction() {
        // Public explorer lookup; unknown identifiers are a valid (and the cheapest) case.
        await this.sutAdapter.sendRequests({
            contractId: 'submitchain', contractFunction: 'VerifyExists',
            contractArguments: [crypto.randomBytes(32).toString('hex')], readOnly: true,
        });
    }
}
module.exports.createWorkloadModule = () => new Query();
