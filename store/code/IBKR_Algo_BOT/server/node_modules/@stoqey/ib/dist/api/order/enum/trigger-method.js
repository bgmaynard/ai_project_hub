"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TriggerMethod = void 0;
/**
 * [[PriceCondition]] trigger method.
 */
var TriggerMethod;
(function (TriggerMethod) {
    TriggerMethod[TriggerMethod["Default"] = 0] = "Default";
    TriggerMethod[TriggerMethod["DoubleBidAsk"] = 1] = "DoubleBidAsk";
    TriggerMethod[TriggerMethod["Last"] = 2] = "Last";
    TriggerMethod[TriggerMethod["DoubleLast"] = 3] = "DoubleLast";
    TriggerMethod[TriggerMethod["BidAsk"] = 4] = "BidAsk";
    TriggerMethod[TriggerMethod["LastOfBidAsk"] = 7] = "LastOfBidAsk";
    TriggerMethod[TriggerMethod["MidPoint"] = 8] = "MidPoint";
})(TriggerMethod || (exports.TriggerMethod = TriggerMethod = {}));
exports.default = TriggerMethod;
//# sourceMappingURL=trigger-method.js.map