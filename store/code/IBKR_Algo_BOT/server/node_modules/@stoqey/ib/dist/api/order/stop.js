"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StopOrder = void 0;
const orderType_1 = require("./enum/orderType");
const tif_1 = require("./enum/tif");
/**
 * Represents a stop order.
 */
class StopOrder {
    constructor(action, auxPrice, totalQuantity, transmit, parentId, tif) {
        this.action = action;
        this.auxPrice = auxPrice;
        this.totalQuantity = totalQuantity;
        this.transmit = transmit;
        this.parentId = parentId;
        this.tif = tif;
        this.orderType = orderType_1.OrderType.STP;
        this.transmit = this.transmit ?? true;
        this.parentId = this.parentId ?? 0;
        this.tif = this.tif ?? tif_1.TimeInForce.DAY;
    }
}
exports.StopOrder = StopOrder;
exports.default = StopOrder;
//# sourceMappingURL=stop.js.map