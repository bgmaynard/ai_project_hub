"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TimeCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * TODO document
 */
class TimeCondition {
    /**
     * Create a [[TimeCondition]] object.
     *
     * @param isMore Before or after...
     * @param time this time... (Valid format: "YYYYMMDD HH:MM:SS")
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(time, isMore, conjunctionConnection) {
        this.time = time;
        this.isMore = isMore;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.Time;
    }
    get strValue() {
        return this.time;
    }
}
exports.TimeCondition = TimeCondition;
exports.default = TimeCondition;
//# sourceMappingURL=time-condition.js.map