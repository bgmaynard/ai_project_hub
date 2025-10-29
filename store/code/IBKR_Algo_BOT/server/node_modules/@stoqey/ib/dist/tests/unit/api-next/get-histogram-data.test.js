"use strict";
/**
 * This file implements tests for the [[IBApiNext.getHistogramData]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const duration_unit_1 = __importDefault(require("../../../api/data/enum/duration-unit"));
describe("RxJS Wrapper: getHistogramData()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.histogramData and verify RxJS result
        const refData = [
            { price: 1, size: 2 },
            { price: 11, size: 12 },
            { price: 21, size: 22 },
        ];
        apiNext
            .getHistogramData({}, true, 1, duration_unit_1.default.DAY)
            .then((data) => {
            expect(data.length).toEqual(refData.length);
            refData.forEach((r, i) => {
                expect(r.price).toEqual(refData[i].price);
                expect(r.size).toEqual(refData[i].size);
            });
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.histogramData, 1, refData);
    });
});
//# sourceMappingURL=get-histogram-data.test.js.map