import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import { OperatorCondition } from "./operator-condition";
/**
 * TODO document
 */
export declare class MarginCondition implements OperatorCondition {
    percent: number;
    isMore: boolean;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
    /**
     * Create a [[MarginCondition]] object.
     *
     * @param isMore If margin is above/below
     * @param percent given percent
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(percent: number, isMore: boolean, conjunctionConnection: ConjunctionConnection);
    get strValue(): string;
}
export default MarginCondition;
