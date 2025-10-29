import SecType from "../../data/enum/sec-type";
import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import OrderCondition from "./order-condition";
/**
 * This class represents a condition requiring a specific execution event to be fulfilled.
 *
 * Orders can be activated or canceled if a set of given conditions is met.
 * An ExecutionCondition is met whenever a trade occurs on a certain product at the given exchange.
 */
export declare class ExecutionCondition implements OrderCondition {
    exchange: string;
    secType: SecType;
    symbol: string;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
    /**
     * Create a [[ExecutionCondition]] object.
     *
     * @param symbol When an execution on symbol
     * @param exchange at exchange
     * @param secType for this secType
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(exchange: string, secType: SecType, symbol: string, conjunctionConnection: ConjunctionConnection);
}
export default ExecutionCondition;
