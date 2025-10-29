"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarketOrder = void 0;
const orderType_1 = require("./enum/orderType");
/**
 * Represents a limit order.
 */
class MarketOrder {
    constructor(action, totalQuantity, transmit, goodAfterTime, goodTillDate) {
        this.action = action;
        this.totalQuantity = totalQuantity;
        this.transmit = transmit;
        this.goodAfterTime = goodAfterTime;
        this.goodTillDate = goodTillDate;
        this.orderType = orderType_1.OrderType.MKT;
        this.transmit = this.transmit ?? true;
        this.goodAfterTime = this.goodAfterTime ?? "";
        this.goodTillDate = this.goodTillDate ?? "";
    }
}
exports.MarketOrder = MarketOrder;
exports.default = MarketOrder;
//# sourceMappingURL=market.js.map