import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import { OperatorCondition } from "./operator-condition";
/**
 * TODO document
 */
export declare class TimeCondition implements OperatorCondition {
    time: string;
    isMore: boolean;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
    /**
     * Create a [[TimeCondition]] object.
     *
     * @param isMore Before or after...
     * @param time this time... (Valid format: "YYYYMMDD HH:MM:SS")
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(time: string, isMore: boolean, conjunctionConnection: ConjunctionConnection);
    get strValue(): string;
}
export default TimeCondition;
