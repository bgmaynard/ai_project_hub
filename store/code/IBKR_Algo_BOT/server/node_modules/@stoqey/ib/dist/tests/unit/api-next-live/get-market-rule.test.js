"use strict";
/**
 * This file implements tests for the [[IBApiNext.getMarketRule]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const logger_1 = __importDefault(require("../../../common/logger"));
describe("ApiNext: getMarketRule()", () => {
    jest.setTimeout(5_000);
    let clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    const api = new __1.IBApiNext();
    const _error$ = api.errorSubject.subscribe((error) => {
        if ((0, __1.isNonFatalError)(error.code, error.error)) {
            logger_1.default.warn(`${error.error.message} (Error #${error.code})`);
        }
        else {
            logger_1.default.error(`${error.error.message} (Error #${error.code}) ${error.advancedOrderReject ? error.advancedOrderReject : ""}`);
        }
    });
    beforeEach(() => {
        api.connect(clientId++);
    });
    afterEach(() => {
        api.disconnect();
    });
    test("getMarketRule once", (done) => {
        const p = [];
        p.push(api.getMarketRule(26));
        p.push(api.getMarketRule(32));
        p.push(api.getMarketRule(67));
        p.push(api.getMarketRule(635));
        p.push(api.getMarketRule(2806));
        Promise.all(p).then((result) => {
            expect(result.length).toBe(5);
            expect(result[0][0].lowEdge).toBe(0);
            expect(result[0][0].increment).toBe(0.01);
            expect(result[1][0].lowEdge).toBe(0);
            expect(result[1][0].increment).toBe(0.01);
            expect(result[2][0].lowEdge).toBe(0);
            expect(result[2][0].increment).toBe(0.25);
            expect(result[3][0].lowEdge).toBe(0);
            expect(result[3][0].increment).toBe(0.0001);
            expect(result[4][0].lowEdge).toBe(0);
            expect(result[4][0].increment).toBe(0.25);
            // logger.info(result);
            done();
        });
    });
    test("getMarketRule n times", (done) => {
        const n = 10;
        const p = [];
        for (let i = 0; i < n; i++)
            p.push(api.getMarketRule(26));
        Promise.all(p).then((result) => {
            // logger.info(result);
            expect(result.length).toBe(n);
            done();
        });
    });
});
//# sourceMappingURL=get-market-rule.test.js.map