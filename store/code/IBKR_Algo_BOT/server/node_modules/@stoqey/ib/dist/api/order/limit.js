"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LimitOrder = void 0;
const orderType_1 = require("./enum/orderType");
/**
 * Represents a limit order.
 */
class LimitOrder {
    constructor(action, lmtPrice, totalQuantity, transmit = true) {
        this.action = action;
        this.lmtPrice = lmtPrice;
        this.totalQuantity = totalQuantity;
        this.transmit = transmit;
        this.orderType = orderType_1.OrderType.LMT;
    }
}
exports.LimitOrder = LimitOrder;
exports.default = LimitOrder;
//# sourceMappingURL=limit.js.map