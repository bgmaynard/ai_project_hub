"use strict";
/**
 * This file implements tests for the [[IBApiNext.getHeadTimestamp]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getHeadTimestamp()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.headTimestamp and verify RxJS result
        const testValue = Math.random();
        apiNext
            .getHeadTimestamp({}, __1.WhatToShow.TRADES, true, 1)
            .then((time) => {
            expect(time).toEqual(testValue);
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.headTimestamp, 1, testValue);
    });
});
//# sourceMappingURL=get-head-timestamp.test.js.map