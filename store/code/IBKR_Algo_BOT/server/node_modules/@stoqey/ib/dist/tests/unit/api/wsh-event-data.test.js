"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
const logger_1 = __importDefault(require("../../../common/logger"));
describe("IBApi Fundamental Data", () => {
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
    test("reqWshMetaData", (done) => {
        const refId = 1;
        ib.once(__1.EventName.connected, () => {
            ib.reqWshMetaData(refId);
        })
            .on(__1.EventName.wshMetaData, (reqId, dataJson) => {
            expect(reqId).toEqual(refId);
            console.log(dataJson);
            ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            const msg = `[${reqId}] ${err.message} (#${code})`;
            if (code == __1.ErrorCode.NEWS_FEED_NOT_ALLOWED) {
                // Ignore this error for tests
                logger_1.default.warn(msg);
                done();
            }
            else if (reqId == refId) {
                done(msg);
            }
        });
        ib.connect();
    });
    test("reqWshEventData deprecated", (done) => {
        const refId = 2;
        ib.once(__1.EventName.connected, () => {
            ib.reqWshEventData(refId, 8314);
        })
            .on(__1.EventName.wshEventData, (reqId, dataJson) => {
            expect(reqId).toEqual(refId);
            console.log(dataJson);
        })
            .on(__1.EventName.scannerDataEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (ib)
                ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            const msg = `[${reqId}] ${err.message} (#${code})`;
            if (code == __1.ErrorCode.NEWS_FEED_NOT_ALLOWED) {
                // Ignore this error for tests
                logger_1.default.warn(msg);
                done();
            }
            else if (reqId == refId) {
                done(msg);
            }
        });
        ib.connect();
    });
    test("reqWshEventData", (done) => {
        const refId = 3;
        ib.once(__1.EventName.connected, () => {
            ib.reqWshEventData(refId, new __1.WshEventData(8314, false, false, false, "20220511", "", 5));
        })
            .on(__1.EventName.wshEventData, (reqId, dataJson) => {
            expect(reqId).toEqual(refId);
            console.log(dataJson);
        })
            .on(__1.EventName.scannerDataEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (ib)
                ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            const msg = `[${reqId}] ${err.message} (#${code})`;
            if (code == __1.ErrorCode.NEWS_FEED_NOT_ALLOWED) {
                // Ignore this error for tests
                logger_1.default.warn(msg);
                done();
            }
            else if (reqId == refId) {
                done(msg);
            }
        });
        ib.connect();
    });
});
//# sourceMappingURL=wsh-event-data.test.js.map