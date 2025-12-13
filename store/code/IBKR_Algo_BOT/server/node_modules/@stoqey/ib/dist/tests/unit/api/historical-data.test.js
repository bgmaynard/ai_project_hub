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
const logger_1 = __importDefault(require("../../../common/logger"));
const contracts_1 = require("../sample-data/contracts");
describe("IBApi Historical data Tests", () => {
    jest.setTimeout(10 * 1000);
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
    it("Stock market data", (done) => {
        const refId = 46;
        let counter = 0;
        ib.once(__1.EventName.connected, () => {
            ib.reqHistoricalData(refId, contracts_1.sample_etf, "20231006-20:00:00", "30 S", __1.BarSizeSetting.SECONDS_ONE, __1.WhatToShow.TRADES, 0, 2, false);
        }).on(__1.EventName.historicalData, (reqId, time, open, high, low, close, volume, count, WAP) => {
            //   console.log(
            //     counter,
            //     time,
            //     open,
            //     high,
            //     low,
            //     close,
            //     volume,
            //     count,
            //     WAP,
            //   );
            expect(reqId).toEqual(refId);
            if (time.startsWith("finished")) {
                expect(counter).toEqual(30);
                done();
            }
            else if (counter++ == 29) {
                expect(time).toEqual("1696622399");
                expect(open).toEqual(429.5);
                expect(high).toEqual(429.6);
                expect(low).toEqual(429.47);
                expect(close).toEqual(429.51);
                expect(volume).toEqual(3487.38);
                expect(count).toEqual(1090);
                expect(WAP).toEqual(429.532);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
    test("Option market data", (done) => {
        const refId = 47;
        let counter = 0;
        ib.once(__1.EventName.connected, () => {
            const contract = new __1.Option("AAPL", "20251219", 200, __1.OptionType.Put);
            ib.reqHistoricalData(refId, contract, "20241107-17:00:00", "30 S", __1.BarSizeSetting.SECONDS_FIFTEEN, __1.WhatToShow.BID_ASK, 0, 2, false);
        }).on(__1.EventName.historicalData, (reqId, time, open, high, low, close, volume, count, WAP) => {
            // console.log(counter, time, open, high, low, close, volume, count, WAP);
            expect(reqId).toEqual(refId);
            if (time.startsWith("finished")) {
                expect(counter).toEqual(2);
                done();
            }
            else if (counter++ == 1) {
                expect(time).toEqual("1730998785");
                expect(open).toEqual(8.77);
                expect(high).toEqual(8.77);
                expect(low).toEqual(8.75);
                expect(close).toEqual(8.77);
                expect(volume).toEqual(-1);
                expect(count).toEqual(-1);
                expect(WAP).toEqual(-1);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
    it("Weekly market data", (done) => {
        const refId = 48;
        let counter = 0;
        ib.once(__1.EventName.connected, () => {
            ib.reqHistoricalData(refId, contracts_1.sample_etf, "20230904-20:00:00", "1 M", __1.BarSizeSetting.WEEKS_ONE, __1.WhatToShow.TRADES, 0, 2, false);
        }).on(__1.EventName.historicalData, (reqId, time, open, high, low, close, volume, count, WAP) => {
            // console.log(
            //   counter,
            //   time,
            //   open,
            //   high,
            //   low,
            //   close,
            //   volume,
            //   count,
            //   WAP,
            // );
            expect(reqId).toEqual(refId);
            if (time.startsWith("finished")) {
                expect(counter).toEqual(5);
                done();
            }
            else if (counter++ == 4) {
                expect(time).toEqual("20230901");
                expect(open).toEqual(437.3);
                expect(high).toEqual(453.67);
                expect(low).toEqual(437.3);
                expect(close).toEqual(450.92);
                expect(volume).toEqual(2771783.24);
                expect(count).toEqual(1393264);
                expect(WAP).toEqual(448.476);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
    it("Monthly market data", (done) => {
        const refId = 49;
        let counter = 0;
        ib.once(__1.EventName.connected, () => {
            ib.reqHistoricalData(refId, contracts_1.sample_etf, "20230904-20:00:00", "1 Y", __1.BarSizeSetting.MONTHS_ONE, __1.WhatToShow.TRADES, 0, 2, false);
        }).on(__1.EventName.historicalData, (reqId, time, open, high, low, close, volume, count, WAP) => {
            // console.log(
            //   counter,
            //   time,
            //   open,
            //   high,
            //   low,
            //   close,
            //   volume,
            //   count,
            //   WAP,
            // );
            expect(reqId).toEqual(refId);
            if (time.startsWith("finished")) {
                expect(counter).toEqual(13);
                done();
            }
            else if (counter++ == 12) {
                expect(time).toEqual("20230901");
                expect(open).toEqual(451.53);
                expect(high).toEqual(453.67);
                expect(low).toEqual(449.68);
                expect(close).toEqual(450.92);
                expect(volume).toEqual(474058.9);
                expect(count).toEqual(248346);
                expect(WAP).toEqual(451.3);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
    it("Test request tick history", (done) => {
        const refId = 45;
        ib.on(__1.EventName.connected, () => {
            ib.reqHistoricalTicks(refId, contracts_1.sample_etf, "20240508-17:00:00", null, 10, __1.WhatToShow.TRADES, 0, true);
        }).on(__1.EventName.historicalTicksLast, (reqId, ticks) => {
            expect(ticks.length).toBeGreaterThan(0);
            done();
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
});
//# sourceMappingURL=historical-data.test.js.map