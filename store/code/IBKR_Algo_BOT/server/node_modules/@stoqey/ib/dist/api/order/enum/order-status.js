"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderStatus = void 0;
/**
 * Order status.
 */
var OrderStatus;
(function (OrderStatus) {
    OrderStatus["ApiPending"] = "ApiPending";
    OrderStatus["ApiCancelled"] = "ApiCancelled";
    OrderStatus["PreSubmitted"] = "PreSubmitted";
    OrderStatus["PendingCancel"] = "PendingCancel";
    OrderStatus["Cancelled"] = "Cancelled";
    OrderStatus["Submitted"] = "Submitted";
    OrderStatus["Filled"] = "Filled";
    OrderStatus["Inactive"] = "Inactive";
    OrderStatus["PendingSubmit"] = "PendingSubmit";
    OrderStatus["Unknown"] = "Unknown";
})(OrderStatus || (exports.OrderStatus = OrderStatus = {}));
exports.default = OrderStatus;
//# sourceMappingURL=order-status.js.map