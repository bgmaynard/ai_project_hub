import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import { TriggerMethod } from "../enum/trigger-method";
import ContractCondition from "./contract-condition";
/**
 * Used with conditional orders to cancel or submit order based on price of an instrument.
 */
export declare class PriceCondition implements ContractCondition {
    price: number;
    triggerMethod: TriggerMethod;
    conId: number;
    exchange: string;
    isMore: boolean;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
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
    constructor(price: number, triggerMethod: TriggerMethod, conId: number, exchange: string, isMore: boolean, conjunctionConnection: ConjunctionConnection);
    get strValue(): string;
}
export default PriceCondition;
