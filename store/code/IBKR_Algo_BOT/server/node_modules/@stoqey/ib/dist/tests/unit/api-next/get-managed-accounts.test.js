"use strict";
/**
 * This file implements tests for the [[IBApiNext.getManagedAccounts]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getManagedAccounts()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.managedAccounts and verify RxJS result
        apiNext
            .getManagedAccounts()
            .then((accounts) => {
            expect(accounts.length).toEqual(3);
            expect(accounts.find((v) => v === "U123456")).toEqual("U123456");
            expect(accounts.find((v) => v === "U987654")).toEqual("U987654");
            expect(accounts.find((v) => v === "U000000")).toEqual("U000000");
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.managedAccounts, "U123456,U987654,U000000");
    });
});
//# sourceMappingURL=get-managed-accounts.test.js.map