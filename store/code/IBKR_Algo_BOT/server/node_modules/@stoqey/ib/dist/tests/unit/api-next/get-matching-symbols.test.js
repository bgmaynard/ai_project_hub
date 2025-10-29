"use strict";
/**
 * This file implements tests for the [[IBApiNext.getMatchingSymbols]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const assert_1 = require("assert");
const __1 = require("../../..");
describe("RxJS Wrapper: searchContracts()", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a error event and verify RxJS result
        const testValue = "We want this error";
        apiNext
            .getMatchingSymbols("AAPL")
            .then(() => {
            (0, assert_1.fail)();
        })
            .catch((e) => {
            expect(e.error.message).toEqual(testValue);
            done();
        });
        api.emit(__1.EventName.error, new Error(testValue), -1, 1);
    });
    test("Result Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a result event and verify RxJS result
        const testValues = [
            {
                contract: {
                    conId: Math.random(),
                },
            },
            {
                contract: {
                    conId: Math.random(),
                },
            },
        ];
        apiNext
            .getMatchingSymbols("AAPL")
            .then((result) => {
            expect(result).toEqual(testValues);
            done();
        })
            .catch((e) => {
            (0, assert_1.fail)(e.error.message);
        });
        api.emit(__1.EventName.symbolSamples, 1, testValues);
    });
});
//# sourceMappingURL=get-matching-symbols.test.js.map