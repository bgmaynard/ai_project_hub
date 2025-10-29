import OrderAction from "./enum/order-action";
import { OrderType } from "./enum/orderType";
import { TimeInForce } from "./enum/tif";
/**
 * Represents a stop-limit order.
 */
export declare class StopLimitOrder {
    action: OrderAction;
    lmtPrice: number;
    auxPrice: number;
    totalQuantity?: boolean;
    transmit?: boolean;
    parentId?: number;
    tif?: TimeInForce;
    constructor(action: OrderAction, lmtPrice: number, auxPrice: number, totalQuantity?: boolean, transmit?: boolean, parentId?: number, tif?: TimeInForce);
    orderType: OrderType;
}
export default StopLimitOrder;
