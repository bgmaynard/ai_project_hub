"use strict";
/**
 * This file implements tests for the [[IBApiNext.getMarketData]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const operators_1 = require("rxjs/operators");
const __1 = require("../../..");
const api_next_1 = require("../../../api-next");
const tickType_1 = __importDefault(require("../../../api/market/tickType"));
describe("RxJS Wrapper: getMarketData()", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a error event and verify RxJS result
        const testValue = "We want this error";
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
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
    test("tickPrice events", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a tickPrice events and verify RxJS result
        const testValueBid = Math.random();
        const testValueAsk = Math.random();
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (data) => {
                switch (data.all.size) {
                    case 2:
                        expect(data.all.get(tickType_1.default.ASK).value).toEqual(testValueAsk);
                    // not break my intention
                    case 1:
                        expect(data.all.get(tickType_1.default.BID).value).toEqual(testValueBid);
                        break;
                }
                if (data.all.size == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.tickPrice, 1, tickType_1.default.BID, testValueBid);
        api.emit(__1.EventName.tickPrice, 1, tickType_1.default.ASK, testValueAsk);
    });
    test("tickSize events", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a tickSize events and verify RxJS result
        const testValueBid = Math.random();
        const testValueAsk = Math.random();
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (data) => {
                switch (data.all.size) {
                    case 2:
                        expect(data.all.get(tickType_1.default.ASK_SIZE).value).toEqual(testValueAsk);
                    // not break my intention
                    case 1:
                        expect(data.all.get(tickType_1.default.BID_SIZE).value).toEqual(testValueBid);
                        break;
                }
                if (data.all.size == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.tickSize, 1, tickType_1.default.BID_SIZE, testValueBid);
        api.emit(__1.EventName.tickSize, 1, tickType_1.default.ASK_SIZE, testValueAsk);
    });
    test("tickGeneric events", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a tickGeneric events and verify RxJS result
        const testValue0 = 12345;
        const testValue1 = 54321;
        let received = 0;
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (data) => {
                if (received == 0) {
                    expect(data.added).toBeDefined();
                    expect(data.changed).toBeUndefined();
                }
                else {
                    expect(data.added).toBeUndefined();
                    expect(data.changed).toBeDefined();
                }
                expect(data.all.get(tickType_1.default.NEWS_TICK).value).toEqual(received ? testValue1 : testValue0);
                received++;
                if (received == 2) {
                    done();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.tickGeneric, 1, tickType_1.default.NEWS_TICK, testValue0);
        api.emit(__1.EventName.tickGeneric, 1, tickType_1.default.NEWS_TICK, testValue1);
    });
    test("tickOptionComputationHandler events", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a tickOptionComputationHandler events and verify RxJS result
        const impliedVolatility = 1;
        const delta = 2;
        const optPrice = 3;
        const pvDividend = 4;
        const gamma = 5;
        const vega = 6;
        const theta = 7;
        const undPrice = 8;
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (data) => {
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_IV).value).toEqual(impliedVolatility);
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_DELTA).value).toEqual(delta);
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_PRICE).value).toEqual(optPrice);
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_GAMMA).value).toEqual(gamma);
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_VEGA).value).toEqual(vega);
                expect(data.added.get(api_next_1.IBApiNextTickType.BID_OPTION_THETA).value).toEqual(theta);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_IV).value).toEqual(impliedVolatility);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_DELTA).value).toEqual(delta);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_PRICE).value).toEqual(optPrice);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_GAMMA).value).toEqual(gamma);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_VEGA).value).toEqual(vega);
                expect(data.all.get(api_next_1.IBApiNextTickType.BID_OPTION_THETA).value).toEqual(theta);
                done();
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.tickOptionComputation, 1, api_next_1.IBApiTickType.BID_OPTION, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice);
    });
    test("Initial value replay to late observers", (done) => {
        // create IBApiNext and reqId counter
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a tickPrice events and verify RxJS result
        let testValue = 1;
        apiNext
            .getMarketData({ conId: 12345 }, null, false, false)
            .pipe((0, operators_1.take)(1))
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: () => {
                apiNext
                    .getMarketData({ conId: 12345 }, null, false, false)
                    // eslint-disable-next-line rxjs/no-ignored-subscription
                    .subscribe({
                    next: (data) => {
                        expect(data.all.get(tickType_1.default.BID).value).toEqual(testValue);
                        if (testValue == 1) {
                            expect(data.added.get(tickType_1.default.BID).value).toEqual(testValue);
                            expect(data.changed).toBeUndefined();
                        }
                        else if (testValue == 2) {
                            expect(data.added).toBeUndefined();
                            expect(data.changed.get(tickType_1.default.BID).value).toEqual(testValue);
                            done();
                            return;
                        }
                        else {
                            fail();
                        }
                        testValue = 2;
                        api.emit(__1.EventName.tickPrice, 1, tickType_1.default.BID, testValue);
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
        api.emit(__1.EventName.tickPrice, 1, tickType_1.default.BID, testValue);
    });
});
//# sourceMappingURL=get-market-data.test.js.map