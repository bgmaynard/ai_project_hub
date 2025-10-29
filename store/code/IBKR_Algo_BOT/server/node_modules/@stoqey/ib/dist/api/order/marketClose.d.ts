import OrderAction from "./enum/order-action";
import { OrderType } from "./enum/orderType";
/**
 * Represents a limit order.
 */
export declare class MarketCloseOrder {
    action: OrderAction;
    totalQuantity: number;
    transmit: boolean;
    constructor(action: OrderAction, totalQuantity: number, transmit?: boolean);
    orderType: OrderType;
}
export default MarketCloseOrder;
