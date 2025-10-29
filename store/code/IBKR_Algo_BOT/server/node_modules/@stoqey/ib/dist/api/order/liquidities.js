"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Liquidities = void 0;
/**
 * Type describing the liquidity type of an execution
 */
var Liquidities;
(function (Liquidities) {
    Liquidities[Liquidities["None"] = 0] = "None";
    Liquidities[Liquidities["Added"] = 1] = "Added";
    Liquidities[Liquidities["Removed"] = 2] = "Removed";
    Liquidities[Liquidities["RoudedOut"] = 3] = "RoudedOut";
})(Liquidities || (exports.Liquidities = Liquidities = {}));
//# sourceMappingURL=liquidities.js.map