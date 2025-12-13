"use strict";
/**
 * This file implements tests for the [[IBApiNext.getContractDetails]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getContractDetails()", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a error event and verify RxJS result
        const testValue = "We want this error";
        apiNext
            .getContractDetails({})
            .then(() => done("failed, then should not be called!"))
            .catch((error) => {
            expect(error.error.message).toEqual(testValue);
            done();
        });
        api.emit(__1.EventName.error, new Error(testValue), -1, 1);
    });
    test("Incremental collection", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit contractDetails and contractDetailsEnd event and verify all subscribers receive it
        const testValue1 = "testValue1";
        const testValue2 = "testValue2";
        apiNext
            .getContractDetails({})
            .then((update) => {
            expect(update.length).toEqual(2);
            switch (update.length) {
                case 2:
                    expect(update[1].marketName).toEqual(testValue2);
                // no break by intention
                case 1:
                    expect(update[0].marketName).toEqual(testValue1);
                    break;
            }
            done();
        })
            .catch((e) => {
            fail(e);
        });
        api.emit(__1.EventName.contractDetails, 1, {
            marketName: testValue1,
        });
        api.emit(__1.EventName.contractDetails, 1, {
            marketName: testValue2,
        });
        api.emit(__1.EventName.contractDetailsEnd, 1);
    });
});
//# sourceMappingURL=get-contract-details.test.js.map