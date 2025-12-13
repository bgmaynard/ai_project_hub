"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutionCondition = void 0;
const order_condition_type_1 = require("../enum/order-condition-type");
/**
 * This class represents a condition requiring a specific execution event to be fulfilled.
 *
 * Orders can be activated or canceled if a set of given conditions is met.
 * An ExecutionCondition is met whenever a trade occurs on a certain product at the given exchange.
 */
class ExecutionCondition {
    /**
     * Create a [[ExecutionCondition]] object.
     *
     * @param symbol When an execution on symbol
     * @param exchange at exchange
     * @param secType for this secType
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(exchange, secType, symbol, conjunctionConnection) {
        this.exchange = exchange;
        this.secType = secType;
        this.symbol = symbol;
        this.conjunctionConnection = conjunctionConnection;
        this.type = order_condition_type_1.OrderConditionType.Execution;
    }
}
exports.ExecutionCondition = ExecutionCondition;
exports.default = ExecutionCondition;
//# sourceMappingURL=execution-condition.js.map