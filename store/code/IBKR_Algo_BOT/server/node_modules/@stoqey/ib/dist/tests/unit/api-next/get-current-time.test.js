"use strict";
/**
 * This file implements tests for the [[IBApiNext.getCurrentTime]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getCurrentTime()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.currentTime and verify RxJS result
        const testValue = Math.random();
        apiNext
            .getCurrentTime()
            .then((time) => {
            expect(time).toEqual(testValue);
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.currentTime, testValue);
    });
});
//# sourceMappingURL=get-current-time.test.js.map