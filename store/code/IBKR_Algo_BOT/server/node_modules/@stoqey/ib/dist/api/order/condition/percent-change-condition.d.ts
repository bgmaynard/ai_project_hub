import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import ContractCondition from "./contract-condition";
/**
 * Used with conditional orders to place or submit an order based on a percentage change of an instrument to the last close price.
 */
export declare class PercentChangeCondition implements ContractCondition {
    percent: number;
    conId: number;
    exchange: string;
    isMore: boolean;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
    /**
     * Create a [[PercentChangeCondition]] object.
     *
     * @param isMore If there is a price percent change measured against last close price above or below...
     * @param percent this amount...
     * @param conId on this contract
     * @param exchange when traded on this exchange...
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(percent: number, conId: number, exchange: string, isMore: boolean, conjunctionConnection: ConjunctionConnection);
    get strValue(): string;
}
export default PercentChangeCondition;
