"use strict";
/**
 * This file implements tests for the [[IBApiNext.getCurrentTime]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getNextValidOrderId()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.nextValidId and verify RxJS result
        const testValue = Math.ceil(Math.random() * 10_000);
        apiNext
            .getNextValidOrderId()
            .then((time) => {
            expect(time).toEqual(testValue);
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.nextValidId, testValue);
    });
});
//# sourceMappingURL=next-valid-order-id.test.js.map