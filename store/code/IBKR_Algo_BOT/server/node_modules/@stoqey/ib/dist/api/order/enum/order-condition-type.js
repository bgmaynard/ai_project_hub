"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderConditionType = void 0;
/**
 * Order condition types.
 */
var OrderConditionType;
(function (OrderConditionType) {
    OrderConditionType[OrderConditionType["Price"] = 1] = "Price";
    OrderConditionType[OrderConditionType["Time"] = 3] = "Time";
    OrderConditionType[OrderConditionType["Margin"] = 4] = "Margin";
    OrderConditionType[OrderConditionType["Execution"] = 5] = "Execution";
    OrderConditionType[OrderConditionType["Volume"] = 6] = "Volume";
    OrderConditionType[OrderConditionType["PercentChange"] = 7] = "PercentChange";
})(OrderConditionType || (exports.OrderConditionType = OrderConditionType = {}));
exports.default = OrderConditionType;
//# sourceMappingURL=order-condition-type.js.map