"use strict";
/**
 * This file implements tests for the [[IBApiNext.getTickByTickAllLastDataUpdates]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getTickByTickAllLastDataUpdates()", () => {
    test("Observable updates", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit EventName.tickByTickAllLast events and verify RxJS result
        const contract = {
            symbol: "AMZN",
            exchange: "SMART",
            currency: "USD",
            secType: __1.SecType.STK,
        };
        const REF_TICKS = [
            {
                contract,
                time: 1675228123,
                price: 1,
                size: 2,
                tickAttribLast: {
                    pastLimit: false,
                    unreported: false,
                },
                exchange: "EXCHANGE",
                specialConditions: "SPECIAL_CONDITIONS",
            },
            {
                contract,
                time: 1675228124,
                price: 11,
                size: 12,
                tickAttribLast: {
                    pastLimit: false,
                    unreported: false,
                },
                exchange: "EXCHANGE",
                specialConditions: "SPECIAL_CONDITIONS",
            },
            {
                contract,
                time: 1675228125,
                price: 21,
                size: 22,
                tickAttribLast: {
                    pastLimit: false,
                    unreported: false,
                },
                exchange: "EXCHANGE",
                specialConditions: "SPECIAL_CONDITIONS",
            },
        ];
        let updateCount = 0;
        apiNext
            .getTickByTickAllLastDataUpdates(contract, 0, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (update) => {
                expect(update.contract).toEqual(REF_TICKS[updateCount].contract);
                expect(update.time).toEqual(REF_TICKS[updateCount].time);
                expect(update.price).toEqual(REF_TICKS[updateCount].price);
                expect(update.size).toEqual(REF_TICKS[updateCount].size);
                expect(update.tickAttribLast).toEqual(REF_TICKS[updateCount].tickAttribLast);
                expect(update.exchange).toEqual(REF_TICKS[updateCount].exchange);
                expect(update.specialConditions).toEqual(REF_TICKS[updateCount].specialConditions);
                updateCount++;
                if (updateCount >= REF_TICKS.length) {
                    done();
                }
            },
            error: (err) => {
                fail(err.error.message);
            },
        });
        for (let i = 0; i < REF_TICKS.length; i++) {
            api.emit(__1.EventName.tickByTickAllLast, 1, 1, REF_TICKS[i].time, REF_TICKS[i].price, REF_TICKS[i].size, REF_TICKS[i].tickAttribLast, REF_TICKS[i].exchange, REF_TICKS[i].specialConditions);
        }
    });
});
//# sourceMappingURL=get-tick-by-tick-all-last-data-updates.test.js.map