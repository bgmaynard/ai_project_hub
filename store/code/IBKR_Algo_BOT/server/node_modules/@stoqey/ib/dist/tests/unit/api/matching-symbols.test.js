"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
describe("IBApi reqMatchingSymbols Tests", () => {
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
    test("SPY", (done) => {
        const refId = 1;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqMatchingSymbols(refId, "SPY");
        })
            .on(__1.EventName.symbolSamples, (reqId, contractDescriptions) => {
            expect(reqId).toEqual(refId);
            expect(contractDescriptions[0].contract.symbol).toEqual("SPY");
            ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("META", (done) => {
        const refId = 2;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqMatchingSymbols(refId, "META");
        })
            .on(__1.EventName.symbolSamples, (reqId, contractDescriptions) => {
            expect(reqId).toEqual(refId);
            expect(contractDescriptions[0].contract.symbol).toEqual("META");
            ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("AMC", (done) => {
        const refId = 3;
        ib.once(__1.EventName.nextValidId, (_reqId) => {
            ib.reqMatchingSymbols(refId, "AMC");
        })
            .on(__1.EventName.symbolSamples, (reqId, contractDescriptions) => {
            expect(reqId).toEqual(refId);
            expect(contractDescriptions[0].contract.symbol).toEqual("AMC");
            ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
});
//# sourceMappingURL=matching-symbols.test.js.map