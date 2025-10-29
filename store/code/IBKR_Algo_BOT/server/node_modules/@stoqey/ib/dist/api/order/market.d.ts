import OrderAction from "./enum/order-action";
import { OrderType } from "./enum/orderType";
/**
 * Represents a limit order.
 */
export declare class MarketOrder {
    action: OrderAction;
    totalQuantity: number;
    transmit?: boolean;
    goodAfterTime?: string;
    goodTillDate?: string;
    constructor(action: OrderAction, totalQuantity: number, transmit?: boolean, goodAfterTime?: string, goodTillDate?: string);
    orderType: OrderType;
}
export default MarketOrder;
