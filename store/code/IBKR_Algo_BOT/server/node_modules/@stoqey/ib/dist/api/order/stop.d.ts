import OrderAction from "./enum/order-action";
import { OrderType } from "./enum/orderType";
import { TimeInForce } from "./enum/tif";
/**
 * Represents a stop order.
 */
export declare class StopOrder {
    action: OrderAction;
    auxPrice: number;
    totalQuantity: number;
    transmit?: boolean;
    parentId?: number;
    tif?: TimeInForce;
    constructor(action: OrderAction, auxPrice: number, totalQuantity: number, transmit?: boolean, parentId?: number, tif?: TimeInForce);
    orderType: OrderType;
}
export default StopOrder;
