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
describe("IBApi Historical news Tests", () => {
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
    it("Get news providers", (done) => {
        ib.once(__1.EventName.connected, () => {
            ib.reqNewsProviders;
        })
            .on(__1.EventName.newsProviders, (newsProviders) => {
            expect(newsProviders).toBeDefined();
            expect(newsProviders).toBeInstanceOf(Array);
            const firstProvider = newsProviders[0];
            expect(firstProvider).toBeDefined();
            expect(firstProvider.providerCode).toBeDefined();
            expect(firstProvider.providerName).toBeDefined();
            done();
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect()
            .reqNewsProviders();
    });
    it("Get PLTR news data with BRFG", (done) => {
        const refId = 46;
        const contractId = 444857009; // PLTR
        const providerCode = "BRFG";
        let received = false;
        ib.once(__1.EventName.connected, () => {
            ib.reqHistoricalNews(refId, contractId, providerCode, "2025-01-13 00:00:00", "2025-01-14 00:00:00", 10, null);
        })
            .on(__1.EventName.historicalNews, (reqId, time, providerCode, articleId, headline) => {
            expect(reqId).toEqual(refId);
            if (reqId == refId)
                received = true;
            expect(time).toBeDefined();
            expect(providerCode).toBeDefined();
            expect(articleId).toBeDefined();
            expect(headline).toBeDefined();
        })
            .on(__1.EventName.historicalNewsEnd, (reqId) => {
            expect(reqId).toEqual(refId);
            if (received)
                done();
            else
                done("Didn't get any result");
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : done(msg);
        })
            .connect();
    });
});
//# sourceMappingURL=historical-news.test.js.map