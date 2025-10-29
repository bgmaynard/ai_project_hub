"use strict";
/**
 * This file implements tests for the [[IBApiNext.getCurrentTime]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const logger_1 = __importDefault(require("../../../common/logger"));
describe("ApiNext: getCurrentTime()", () => {
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
    test("getCurrentTime once", (done) => {
        api
            .getCurrentTime()
            .then((result) => {
            // logger.info(result);
            done();
        })
            .catch((err) => {
            done(`getCurrentTime failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
    test("getCurrentTime twice", (done) => {
        const p1 = api.getCurrentTime();
        const p2 = api.getCurrentTime();
        Promise.all([p1, p2])
            .then((result) => {
            // logger.info(result[0], result[1]);
            done();
        })
            .catch((err) => {
            done(`getCurrentTime failed with '${err.error?.message}' (Error #${err.code})`);
        });
    });
    test("getCurrentTime n times", (done) => {
        const n = 10;
        const p = [];
        for (let i = 0; i < n; i++)
            p.push(api.getCurrentTime());
        Promise.all(p).then((result) => {
            // logger.info(result);
            expect(result.length).toBe(n);
            done();
        });
    });
});
//# sourceMappingURL=get-current-time.test.js.map