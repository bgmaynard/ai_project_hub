import OrderAction from "./enum/order-action";
import { OrderType } from "./enum/orderType";
/**
 * Represents a limit order.
 */
export declare class LimitOrder {
    action: OrderAction;
    lmtPrice: number;
    totalQuantity: number;
    transmit: boolean;
    constructor(action: OrderAction, lmtPrice: number, totalQuantity: number, transmit?: boolean);
    orderType: OrderType;
}
export default LimitOrder;
