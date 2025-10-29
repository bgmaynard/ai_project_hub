"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.VolumeCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * Used with conditional orders to submit or cancel an order based on a specified volume change in a security.
 */
class VolumeCondition {
    /**
     * Create a [[VolumeCondition]] object.
     *
     * @param conId Whenever contract...
     * @param exchange When traded at
     * @param isMore reaches a volume higher/lower
     * @param volume than this...
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(volume, conId, exchange, isMore, conjunctionConnection) {
        this.volume = volume;
        this.conId = conId;
        this.exchange = exchange;
        this.isMore = isMore;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.Volume;
    }
    get strValue() {
        return "" + this.volume;
    }
}
exports.VolumeCondition = VolumeCondition;
exports.default = VolumeCondition;
//# sourceMappingURL=volume-condition.js.map