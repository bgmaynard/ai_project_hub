"use strict";
/**
 * This file implements tests for the [[IBApiNext.getContractDetails]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const logger_1 = __importDefault(require("../../../common/logger"));
const contracts_1 = require("../sample-data/contracts");
describe("ApiNext: getContractDetails()", () => {
    jest.setTimeout(10 * 1000);
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    let api;
    let error$;
    beforeEach(() => {
        api = new __1.IBApiNext();
        if (!error$) {
            error$ = api.errorSubject.subscribe((error) => {
                if (error.reqId === -1) {
                    logger_1.default.warn(`${error.error.message} (Error #${error.code})`);
                }
                else {
                    logger_1.default.error(`${error.error.message} (Error #${error.code}) ${error.advancedOrderReject ? error.advancedOrderReject : ""}`);
                }
            });
        }
        try {
            api.connect(clientId);
        }
        catch (error) {
            logger_1.default.error(error.message);
        }
    });
    afterEach(() => {
        if (api) {
            api.disconnect();
            api = undefined;
        }
    });
    test("ETF historical ticks last", (done) => {
        api
            .getHistoricalTicksLast(contracts_1.sample_etf, "20240508-17:00:00", null, 1, true)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (ticks) => {
                //   console.log(ticks.length, ticks);
                expect(ticks.length).toEqual(17);
                expect(ticks[0].time).toEqual(1715187600);
                expect(ticks[0].price).toEqual(516.635);
                expect(ticks[0].size).toEqual(1);
                expect(ticks[0].exchange).toEqual("FINRA");
            },
            complete: () => {
                done();
            },
            error: () => {
                done("Some error occured!");
            },
        });
    });
    test("ETF historical bid/ask ticks", (done) => {
        api
            .getHistoricalTicksBidAsk(contracts_1.sample_etf, "20240508-17:00:00", null, 1, true, true)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (ticks) => {
                //   console.log(ticks.length, ticks);
                expect(ticks.length).toEqual(34);
                expect(ticks[0].time).toEqual(1715187599);
                expect(ticks[0].priceBid).toEqual(516.62);
                expect(ticks[0].priceAsk).toEqual(516.63);
                expect(ticks[0].sizeBid).toEqual(1100);
                expect(ticks[0].sizeAsk).toEqual(1400);
            },
            complete: () => {
                done();
            },
            error: () => {
                done("Some error occured!");
            },
        });
    });
});
//# sourceMappingURL=get-historical-ticks.test.js.map