"use strict";
/**
 * This file implements tests for the [[IBApiNext.getHistoricalData]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getHistoricalData()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit EventName.historicalData and verify RxJS result
        const REF_BARS = [
            {
                time: "20210203 01:02:03",
                open: 1,
                high: 2,
                low: 3,
                close: 4,
                volume: 5,
                count: 6,
                WAP: 7,
            },
            {
                time: "20210203 02:02:03",
                open: 11,
                high: 12,
                low: 13,
                close: 14,
                volume: 15,
                count: 16,
                WAP: 17,
            },
            {
                time: "20210203 03:02:03",
                open: 21,
                high: 22,
                low: 23,
                close: 24,
                volume: 25,
                count: 26,
                WAP: 27,
            },
        ];
        apiNext
            .getHistoricalData({}, "", "", "", __1.WhatToShow.None, 0, 1)
            .then((bars) => {
            for (let i = 0; i < REF_BARS.length; i++) {
                expect(bars[i].time).toEqual(REF_BARS[i].time);
                expect(bars[i].open).toEqual(REF_BARS[i].open);
                expect(bars[i].high).toEqual(REF_BARS[i].high);
                expect(bars[i].low).toEqual(REF_BARS[i].low);
                expect(bars[i].close).toEqual(REF_BARS[i].close);
                expect(bars[i].volume).toEqual(REF_BARS[i].volume);
                expect(bars[i].count).toEqual(REF_BARS[i].count);
                expect(bars[i].WAP).toEqual(REF_BARS[i].WAP);
            }
            done();
        })
            .catch((err) => {
            fail(err.error.message);
        });
        for (let i = 0; i < REF_BARS.length; i++) {
            api.emit(__1.EventName.historicalData, 1, REF_BARS[i].time, REF_BARS[i].open, REF_BARS[i].high, REF_BARS[i].low, REF_BARS[i].close, REF_BARS[i].volume, REF_BARS[i].count, REF_BARS[i].WAP);
        }
        api.emit(__1.EventName.historicalData, 1, "finished-", -1, -1, -1, -1, -1, -1, -1);
    });
    test("Promise error", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit EventName.error and verify RxJS result
        apiNext
            .getHistoricalData({}, "", "", "", __1.WhatToShow.None, 0, 1)
            .then(() => {
            fail();
        })
            .catch(() => {
            done();
        });
        api.emit(__1.EventName.error, new Error(), __1.ErrorCode.UNSUPPORTED_VERSION, 1);
    });
});
//# sourceMappingURL=get-historical-data.test.js.map