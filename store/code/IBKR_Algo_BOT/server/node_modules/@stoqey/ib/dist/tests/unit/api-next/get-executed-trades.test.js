"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const event_name_1 = require("../../../api/data/enum/event-name");
const sec_type_1 = __importDefault(require("../../../api/data/enum/sec-type"));
describe("RxJS Wrapper: getExecutionDetails", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        const executedTrades = [
            {
                reqId: 1,
                contract: {
                    conId: 265598,
                    symbol: "AAPL",
                    secType: sec_type_1.default.STK,
                    lastTradeDateOrContractMonth: "",
                    strike: 0,
                    multiplier: 0,
                    exchange: "ARCA",
                    currency: "USD",
                    localSymbol: "AAPL",
                    tradingClass: "NMS",
                },
                execution: {
                    orderId: 9,
                    execId: "0000e0d5.619dbad4.01.01",
                    time: "20210920  17:07:33",
                    acctNumber: "DU3360023",
                    exchange: "ARCA",
                    side: "BOT",
                    shares: 10,
                    price: 143.82,
                    permId: 723015772,
                    clientId: 0,
                    liquidation: 0,
                    cumQty: 10,
                    avgPrice: 143.82,
                    orderRef: "",
                    evRule: "",
                    evMultiplier: 0,
                    modelCode: "",
                    lastLiquidity: __1.Liquidities.Removed,
                },
            },
        ];
        const executionFilter = {};
        const reqId = 1;
        apiNext.getExecutionDetails(executionFilter).then((data) => {
            expect(data.length).toEqual(1);
            expect(data[0]["reqId"]).toMatchObject(executedTrades);
            done();
        });
        api.emit(event_name_1.EventName.execDetails, executedTrades);
        api.emit(event_name_1.EventName.execDetailsEnd, reqId);
    });
});
//# sourceMappingURL=get-executed-trades.test.js.map