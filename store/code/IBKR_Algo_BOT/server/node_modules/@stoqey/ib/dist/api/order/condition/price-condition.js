"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PriceCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * Used with conditional orders to cancel or submit order based on price of an instrument.
 */
class PriceCondition {
    /**
     * Create a [[PriceCondition]] object.
     *
     * @param conId When this contract...
     * @param exchange traded on this exchange
     * @param isMore has a price above/below
     * @param price this quantity
     * @param triggerMethod TODO document
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(price, triggerMethod, conId, exchange, isMore, conjunctionConnection) {
        this.price = price;
        this.triggerMethod = triggerMethod;
        this.conId = conId;
        this.exchange = exchange;
        this.isMore = isMore;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.Price;
    }
    get strValue() {
        return "" + this.price;
    }
}
exports.PriceCondition = PriceCondition;
exports.default = PriceCondition;
//# sourceMappingURL=price-condition.js.map