"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.sample_order = void 0;
const __1 = require("../../..");
exports.sample_order = {
    orderType: __1.OrderType.LMT,
    action: __1.OrderAction.BUY,
    lmtPrice: 1,
    totalQuantity: 1,
    conditionsIgnoreRth: true,
    conditionsCancelOrder: false,
    tif: __1.TimeInForce.DAY,
    transmit: true,
    goodAfterTime: "20300101-01:01:01",
};
//# sourceMappingURL=orders.js.map