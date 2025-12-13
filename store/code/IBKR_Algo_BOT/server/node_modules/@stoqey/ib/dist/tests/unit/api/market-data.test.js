"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * This file implement test code for the public API interfaces.
 */
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
const contracts_1 = require("../sample-data/contracts");
describe("IBApi Market data Tests", () => {
    jest.setTimeout(20_000); // reqMktData requires up te 11 secs to run
    let ib;
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId,
        });
        // logger.info("IBApi created");
    });
    afterEach(() => {
        if (ib) {
            ib.disconnect();
            ib = undefined;
        }
        // logger.info("IBApi disconnected");
    });
    const IsError = (code) => code !== __1.ErrorCode.REQ_MKT_DATA_NOT_AVAIL &&
        code !== __1.ErrorCode.DISPLAYING_DELAYED_DATA;
    it("ETF market data", (done) => {
        const refId = 45;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_etf, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    it("Stock market data", (done) => {
        const refId = 46;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_stock, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    test("Option market data", (done) => {
        const refId = 47;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_option, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    it("Future market data", (done) => {
        const refId = 48;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_future, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    it("DAX market data", (done) => {
        const refId = 49;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_dax_index, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    it("Index market data", (done) => {
        const refId = 50;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_index, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
    it("Crypto market data", (done) => {
        const refId = 51;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqMktData(refId, contracts_1.sample_crypto, "", true, false);
        })
            .on(__1.EventName.tickPrice, (reqId, _field, _value) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            // console.log(_field, _value);
        })
            .on(__1.EventName.tickSnapshotEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (IsError(code))
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqMarketDataType(__1.MarketDataType.DELAYED_FROZEN);
    });
});
//# sourceMappingURL=market-data.test.js.map