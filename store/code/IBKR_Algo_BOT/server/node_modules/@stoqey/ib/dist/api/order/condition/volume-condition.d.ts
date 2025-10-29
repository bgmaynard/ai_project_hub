import { ConjunctionConnection } from "../enum/conjunction-connection";
import { OrderConditionType } from "../enum/order-condition-type";
import ContractCondition from "./contract-condition";
/**
 * Used with conditional orders to submit or cancel an order based on a specified volume change in a security.
 */
export declare class VolumeCondition implements ContractCondition {
    volume: number;
    conId: number;
    exchange: string;
    isMore: boolean;
    conjunctionConnection: ConjunctionConnection;
    type: OrderConditionType;
    /**
     * Create a [[VolumeCondition]] object.
     *
     * @param conId Whenever contract...
     * @param exchange When traded at
     * @param isMore reaches a volume higher/lower
     * @param volume than this...
     * @param conjunctionConnection AND | OR next condition (will be ignored if no more conditions are added)
     */
    constructor(volume: number, conId: number, exchange: string, isMore: boolean, conjunctionConnection: ConjunctionConnection);
    get strValue(): string;
}
export default VolumeCondition;
