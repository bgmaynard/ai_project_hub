"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarketCloseOrder = void 0;
const orderType_1 = require("./enum/orderType");
/**
 * Represents a limit order.
 */
class MarketCloseOrder {
    constructor(action, totalQuantity, transmit = true) {
        this.action = action;
        this.totalQuantity = totalQuantity;
        this.transmit = transmit;
        this.orderType = orderType_1.OrderType.MOC;
    }
}
exports.MarketCloseOrder = MarketCloseOrder;
exports.default = MarketCloseOrder;
//# sourceMappingURL=marketClose.js.map