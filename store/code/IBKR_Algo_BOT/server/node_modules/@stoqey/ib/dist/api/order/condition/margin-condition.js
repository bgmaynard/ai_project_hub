"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarginCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * TODO document
 */
class MarginCondition {
    /**
     * Create a [[MarginCondition]] object.
     *
     * @param isMore If margin is above/below
     * @param percent given percent
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(percent, isMore, conjunctionConnection) {
        this.percent = percent;
        this.isMore = isMore;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.Margin;
    }
    get strValue() {
        return "" + this.percent;
    }
}
exports.MarginCondition = MarginCondition;
exports.default = MarginCondition;
//# sourceMappingURL=margin-condition.js.map