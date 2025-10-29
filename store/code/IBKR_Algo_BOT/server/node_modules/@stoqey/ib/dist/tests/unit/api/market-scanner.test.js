"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
const logger_1 = __importDefault(require("../../../common/logger"));
describe("IBApi market scanner tests", () => {
    jest.setTimeout(10_000);
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
        ib.disconnect();
    });
    test("Scanner parameters", (done) => {
        ib.once(__1.EventName.connected, () => {
            ib.reqScannerParameters();
        }).on(__1.EventName.scannerParameters, (xml) => {
            const match = '<?xml version="1.0" encoding="UTF-8"?>'; // eslint-disable-line quotes
            expect(xml.substring(0, match.length)).toEqual(match);
            done();
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : logger_1.default.error(msg);
        })
            .connect();
    });
    test("Most active US stocks", (done) => {
        const refId = 1;
        ib.once(__1.EventName.connected, () => {
            ib.reqScannerSubscription(refId, {
                abovePrice: 1,
                scanCode: __1.ScanCode.MOST_ACTIVE,
                locationCode: __1.LocationCode.STK_US,
                instrument: __1.Instrument.STK,
                numberOfRows: 20,
            });
        })
            .on(__1.EventName.scannerData, (reqId, _rank, _contract, _distance, _benchmark, _projection, _legStr) => {
            expect(reqId).toEqual(refId);
        })
            .on(__1.EventName.scannerDataEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            done();
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : logger_1.default.error(msg);
        })
            .connect();
    });
});
//# sourceMappingURL=market-scanner.test.js.map