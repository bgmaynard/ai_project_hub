"use strict";
/**
 * This file implements tests for the [[IBApiNext.getPnL]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getPnL()", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a error event and verify RxJS result
        const testValue = "We want this error";
        apiNext
            .getPnL("U123456")
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
    test("Value update", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a pnl event and verify RxJS result
        const dailyPnL = 1234;
        const unrealizedPnL = 56788;
        const realizedPnL = 901234;
        apiNext
            .getPnL("U123456")
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (pnl) => {
                expect(pnl.dailyPnL).toEqual(dailyPnL);
                expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                expect(pnl.realizedPnL).toEqual(realizedPnL);
                done();
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.pnl, 1, dailyPnL, unrealizedPnL, realizedPnL);
    });
    test("Initial value replay to late observers", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a pnl event and verify RxJS result
        const dailyPnL = 1234;
        const unrealizedPnL = 56788;
        const realizedPnL = 901234;
        apiNext
            .getPnL("U123456")
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: () => {
                apiNext
                    .getPnL("U123456")
                    // eslint-disable-next-line rxjs/no-ignored-subscription
                    .subscribe({
                    next: (pnl) => {
                        expect(pnl.dailyPnL).toEqual(dailyPnL);
                        expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                        expect(pnl.realizedPnL).toEqual(realizedPnL);
                        done();
                    },
                    error: (error) => {
                        fail(error.error.message);
                    },
                });
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.pnl, 1, dailyPnL, unrealizedPnL, realizedPnL);
    });
    test("Multi-account with multiple-subscribers", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // testing values
        const accountId1 = "DU123456";
        const accountId2 = "DU123456";
        const dailyPnL = 1234;
        const unrealizedPnL = 56788;
        const realizedPnL = 901234;
        // emit as accountSummary event and verify all subscribers receive it
        let receivedAccount1Updates = 0;
        let receivedAccount2Updates = 0;
        // reqId 2
        apiNext
            .getPnL(accountId1)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (pnl) => {
                expect(pnl.dailyPnL).toEqual(dailyPnL);
                expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                expect(pnl.realizedPnL).toEqual(realizedPnL);
                receivedAccount1Updates++;
                if (receivedAccount1Updates == 2 && receivedAccount2Updates == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        apiNext
            .getPnL(accountId1)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (pnl) => {
                expect(pnl.dailyPnL).toEqual(dailyPnL);
                expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                expect(pnl.realizedPnL).toEqual(realizedPnL);
                receivedAccount1Updates++;
                if (receivedAccount1Updates == 2 && receivedAccount2Updates == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        // reqId 2
        apiNext
            .getPnL(accountId2)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (pnl) => {
                expect(pnl.dailyPnL).toEqual(dailyPnL);
                expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                expect(pnl.realizedPnL).toEqual(realizedPnL);
                receivedAccount2Updates++;
                if (receivedAccount1Updates == 2 && receivedAccount2Updates == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        apiNext
            .getPnL(accountId2)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (pnl) => {
                expect(pnl.dailyPnL).toEqual(dailyPnL);
                expect(pnl.unrealizedPnL).toEqual(unrealizedPnL);
                expect(pnl.realizedPnL).toEqual(realizedPnL);
                receivedAccount2Updates++;
                if (receivedAccount1Updates == 2 && receivedAccount2Updates == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.pnl, 1, dailyPnL, unrealizedPnL, realizedPnL);
        api.emit(__1.EventName.pnl, 2, dailyPnL, unrealizedPnL, realizedPnL);
    });
});
//# sourceMappingURL=get-pnl.test.js.map