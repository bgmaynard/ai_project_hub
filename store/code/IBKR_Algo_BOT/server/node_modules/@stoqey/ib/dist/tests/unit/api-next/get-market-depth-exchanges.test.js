"use strict";
/**
 * This file implements tests for the [[IBApiNext.getMarketDepthExchanges]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
describe("RxJS Wrapper: getMarketDepthExchanges()", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // emit a EventName.mktDepthExchanges and verify RxJS result
        const depthMktDataDescriptions = [
            {
                exchange: "CME",
                secType: __1.SecType.FUT,
            },
            {
                exchange: "Cboe",
                secType: __1.SecType.OPT,
            },
        ];
        apiNext
            .getMarketDepthExchanges()
            .then((data) => {
            expect(data.length).toEqual(2);
            expect(data[0].exchange).toEqual(depthMktDataDescriptions[0].exchange);
            expect(data[0].secType).toEqual(depthMktDataDescriptions[0].secType);
            expect(data[1].exchange).toEqual(depthMktDataDescriptions[1].exchange);
            expect(data[1].secType).toEqual(depthMktDataDescriptions[1].secType);
            done();
        })
            .catch((error) => {
            fail(error.error.message);
        });
        api.emit(__1.EventName.mktDepthExchanges, depthMktDataDescriptions);
    });
});
//# sourceMappingURL=get-market-depth-exchanges.test.js.map