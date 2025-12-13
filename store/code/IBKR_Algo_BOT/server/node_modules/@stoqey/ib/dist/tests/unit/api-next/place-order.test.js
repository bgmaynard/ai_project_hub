"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
describe("Place orders to IB", () => {
    test("Error Event", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        api
            .on(__1.EventName.error, (error, _code, _reqId) => {
            fail(error.message);
        })
            .on(__1.EventName.openOrder, (openOrderId, _contract, _order, _orderState) => {
            expect(openOrderId).toEqual(orderId);
            // done();
        })
            .on(__1.EventName.orderStatus, (openOrderId, status, filled, ..._arg) => {
            expect(openOrderId).toEqual(orderId);
            expect(status).toEqual(__1.OrderStatus.Submitted);
            expect(filled).toBeFalsy();
            // done();
        })
            .on(__1.EventName.openOrderEnd, () => {
            done();
        });
        const orderId = 1;
        const contract = new __1.Stock("SPY");
        const order = {
            orderType: __1.OrderType.LMT,
            action: __1.OrderAction.BUY,
            lmtPrice: 120,
            orderId,
            totalQuantity: 10,
            account: configuration_1.default.ib_test_account,
        };
        apiNext.placeOrder(orderId, contract, order);
        api.emit(__1.EventName.openOrder, orderId, order, contract);
        api.emit(__1.EventName.orderStatus, orderId, __1.OrderStatus.Submitted, 0);
        api.emit(__1.EventName.openOrderEnd);
    });
});
//# sourceMappingURL=place-order.test.js.map