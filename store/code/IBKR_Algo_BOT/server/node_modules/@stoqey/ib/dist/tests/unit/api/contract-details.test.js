"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * This file implements tests for the [[reqContractDetails]] API entry point.
 */
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
const contracts_1 = require("../sample-data/contracts");
describe("IBApi reqContractDetails Tests", () => {
    jest.setTimeout(5000);
    let ib;
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId,
        });
    });
    afterEach(() => {
        if (ib) {
            ib.disconnect();
            ib = undefined;
        }
    });
    test("Forex", (done) => {
        const refId = 1;
        const refContract = new __1.Forex("USD", "EUR");
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqContractDetails(refId, refContract);
        })
            .on(__1.EventName.contractDetails, (reqId, details) => {
            expect(reqId).toEqual(refId);
            expect(details.contract.secType).toEqual(refContract.secType);
            expect(details.contract.symbol).toEqual(refContract.symbol);
            expect(details.contract.currency).toEqual(refContract.currency);
            expect(details.marketName).toEqual(`${refContract.symbol}.${refContract.currency}`);
        })
            .on(__1.EventName.contractDetailsEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("Stock", (done) => {
        const refId = 2;
        const refContract = contracts_1.sample_stock;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqContractDetails(refId, refContract);
        })
            .on(__1.EventName.contractDetails, (reqId, details) => {
            expect(reqId).toEqual(refId);
            expect(details.contract.secType).toEqual(refContract.secType);
            expect(details.contract.symbol).toEqual(refContract.symbol);
            expect(details.contract.currency).toEqual(refContract.currency);
        })
            .on(__1.EventName.contractDetailsEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("Option", (done) => {
        const refId = 3;
        const refContract = contracts_1.sample_option;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqContractDetails(refId, refContract);
        })
            .on(__1.EventName.contractDetails, (reqId, details) => {
            expect(reqId).toEqual(refId);
            expect(details.contract.secType).toEqual(refContract.secType);
            expect(details.contract.symbol).toEqual(refContract.symbol);
            expect(details.contract.currency).toEqual(refContract.currency);
            expect(details.contract.conId).toEqual(653318228);
        })
            .on(__1.EventName.contractDetailsEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("Option chain", (done) => {
        const refId = 4;
        let count = 0;
        const refContract = contracts_1.sample_option;
        refContract.strike = 0;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqContractDetails(refId, refContract);
        })
            .on(__1.EventName.contractDetails, (reqId, details) => {
            expect(reqId).toEqual(refId);
            expect(details.contract.secType).toEqual(refContract.secType);
            expect(details.contract.symbol).toEqual(refContract.symbol);
            expect(details.contract.currency).toEqual(refContract.currency);
            count++;
        })
            .on(__1.EventName.contractDetailsEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            expect(count).toBeGreaterThanOrEqual(92);
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
});
//# sourceMappingURL=contract-details.test.js.map