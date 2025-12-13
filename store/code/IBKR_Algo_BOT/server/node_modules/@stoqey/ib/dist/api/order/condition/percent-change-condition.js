"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PercentChangeCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * Used with conditional orders to place or submit an order based on a percentage change of an instrument to the last close price.
 */
class PercentChangeCondition {
    /**
     * Create a [[PercentChangeCondition]] object.
     *
     * @param isMore If there is a price percent change measured against last close price above or below...
     * @param percent this amount...
     * @param conId on this contract
     * @param exchange when traded on this exchange...
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(percent, conId, exchange, isMore, conjunctionConnection) {
        this.percent = percent;
        this.conId = conId;
        this.exchange = exchange;
        this.isMore = isMore;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.PercentChange;
    }
    get strValue() {
        return "" + this.percent;
    }
}
exports.PercentChangeCondition = PercentChangeCondition;
exports.default = PercentChangeCondition;
//# sourceMappingURL=percent-change-condition.js.map