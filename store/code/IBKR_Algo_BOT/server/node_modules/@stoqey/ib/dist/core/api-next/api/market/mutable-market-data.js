"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MutableMarketDataUpdate = exports.MutableMarketData = void 0;
const item_list_update_1 = require("../../item-list-update");
const map_1 = require("../../map");
/** Mutable version of [[AccountSummary]] */
class MutableMarketData extends map_1.IBApiNextMap {
}
exports.MutableMarketData = MutableMarketData;
/** Mutable version of [[AccountSummariesUpdate]] */
class MutableMarketDataUpdate extends item_list_update_1.IBApiNextItemListUpdate {
}
exports.MutableMarketDataUpdate = MutableMarketDataUpdate;
//# sourceMappingURL=mutable-market-data.js.map