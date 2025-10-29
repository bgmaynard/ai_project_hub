"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TrailingStopOrder = void 0;
const orderType_1 = require("./enum/orderType");
const tif_1 = require("./enum/tif");
/**
 * Represents a trailing-stop order.
 */
class TrailingStopOrder {
    /**
     * Create a trailing-stop order.
     *
     * @param action The order action (buy or sell)
     * @param totalQuantity The number of positions being bought/sold.
     * @param auxPrice Generic field to contain the stop price for STP LMT orders, trailing amount, etc.
     * @param trailingPercent Specifies the trailing amount of a trailing stop order as a percentage.
     * This field is mutually exclusive with the existing trailing amount.
     * That is, the API client can send one or the other but not both.
     * This field is read AFTER the stop price (barrier price) as follows: deltaNeutralAuxPrice stopPrice, trailingPercent, scale order attributes.
     * The field will also be sent to the API in the openOrder message if the API client version is >= 56.
     * It is sent after the stopPrice field as follows: stopPrice, trailingPct, basisPoint
     * @param transmit Specifies whether the order will be transmitted by TWS. If set to false`,
     * the order will be created at TWS but will not be sent.
     * @param parentId The order ID of the parent order, used for bracket and auto trailing stop orders..
     * @param tif  The time in force.
     * Valid values are:
     * - DAY - Valid for the day only.
     * - GTC - Good until canceled.
     */
    constructor(action, totalQuantity, auxPrice, trailingPercent, transmit, parentId, tif) {
        this.action = action;
        this.totalQuantity = totalQuantity;
        this.auxPrice = auxPrice;
        this.trailingPercent = trailingPercent;
        this.transmit = transmit;
        this.parentId = parentId;
        this.tif = tif;
        /** The order's type (must be [[OrderType.TRAIL]]). */
        this.orderType = orderType_1.OrderType.TRAIL;
        this.transmit = this.transmit ?? true;
        this.parentId = this.parentId ?? 0;
        this.tif = this.tif ?? tif_1.TimeInForce.DAY;
    }
}
exports.TrailingStopOrder = TrailingStopOrder;
exports.default = TrailingStopOrder;
//# sourceMappingURL=trailingStop.js.map