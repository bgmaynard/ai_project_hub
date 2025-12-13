"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StopLimitOrder = void 0;
const orderType_1 = require("./enum/orderType");
const tif_1 = require("./enum/tif");
/**
 * Represents a stop-limit order.
 */
class StopLimitOrder {
    constructor(action, lmtPrice, auxPrice, totalQuantity, transmit, parentId, tif) {
        this.action = action;
        this.lmtPrice = lmtPrice;
        this.auxPrice = auxPrice;
        this.totalQuantity = totalQuantity;
        this.transmit = transmit;
        this.parentId = parentId;
        this.tif = tif;
        this.orderType = orderType_1.OrderType.STP_LMT;
        this.transmit = this.transmit ?? true;
        this.parentId = this.parentId ?? 0;
        this.tif = this.tif ?? tif_1.TimeInForce.DAY;
    }
}
exports.StopLimitOrder = StopLimitOrder;
exports.default = StopLimitOrder;
//# sourceMappingURL=stopLimit.js.map