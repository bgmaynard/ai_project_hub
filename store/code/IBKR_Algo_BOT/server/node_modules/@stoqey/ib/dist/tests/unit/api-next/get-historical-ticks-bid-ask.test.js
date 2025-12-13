"use strict";
/**
 * This file implements tests for the [[IBApiNext.getHistoricalTicksBidAsk]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getHistoricalTicksBidAsk()", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a error event and verify RxJS result
        const testValue = "We want this error";
        apiNext
            .getHistoricalTicksBidAsk({}, "", "", 1, 1, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: () => {
                fail();
            },
            error: (error) => {
                expect(error.error.message).toEqual(testValue);
                done();
            },
        });
        api.emit(__1.EventName.error, new Error(testValue), -1, 1);
    });
    test("Incremental collection and complete event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit historicalTicks events and verify all subscribers receive it
        const firstTick = {
            time: 12345,
            priceBid: 101,
            priceAsk: 102,
            sizeBid: 104,
            sizeAsk: 105,
        };
        const secondTick = {
            time: 12346,
            priceBid: 201,
            priceAsk: 202,
            sizeBid: 204,
            sizeAsk: 205,
        };
        apiNext
            .getHistoricalTicksBidAsk({}, "", "", 1, 1, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (ticks) => {
                if (!ticks || !ticks.length || ticks.length > 2) {
                    fail();
                }
                if (ticks.length >= 1) {
                    expect(ticks[0].time).toEqual(firstTick.time);
                    expect(ticks[0].priceBid).toEqual(firstTick.priceBid);
                    expect(ticks[0].priceAsk).toEqual(firstTick.priceAsk);
                    expect(ticks[0].sizeBid).toEqual(firstTick.sizeBid);
                    expect(ticks[0].sizeAsk).toEqual(firstTick.sizeAsk);
                }
                if (ticks.length == 2) {
                    expect(ticks[1].time).toEqual(secondTick.time);
                    expect(ticks[1].priceBid).toEqual(secondTick.priceBid);
                    expect(ticks[1].priceAsk).toEqual(secondTick.priceAsk);
                    expect(ticks[1].sizeBid).toEqual(secondTick.sizeBid);
                    expect(ticks[1].sizeAsk).toEqual(secondTick.sizeAsk);
                }
            },
            complete: () => {
                done();
            },
            error: () => {
                fail();
            },
        });
        api.emit(__1.EventName.historicalTicksBidAsk, 1, [firstTick], false);
        api.emit(__1.EventName.historicalTicksBidAsk, 1, [secondTick], true);
    });
});
//# sourceMappingURL=get-historical-ticks-bid-ask.test.js.map