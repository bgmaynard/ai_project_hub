"use strict";
/**
 * Interactive Brokers Typescript API
 *
 * ````
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Stoqey
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ````
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ScanCode = exports.LocationCode = exports.Instrument = exports.TrailingStopOrder = exports.StopLimitOrder = exports.StopOrder = exports.MarketCloseOrder = exports.MarketOrder = exports.Liquidities = exports.LimitOrder = exports.TriggerMethod = exports.OrderType = exports.OrderStatus = exports.OrderConditionType = exports.OrderAction = exports.ConjunctionConnection = exports.VolumeCondition = exports.TimeCondition = exports.PriceCondition = exports.PercentChangeCondition = exports.MarginCondition = exports.ExecutionCondition = exports.TickByTickDataType = exports.BarSizeSetting = exports.DurationUnit = exports.SecType = exports.OptionType = exports.OptionExerciseAction = exports.MIN_SERVER_VER = exports.LogLevel = exports.FADataType = exports.EventName = exports.WshEventData = exports.Stock = exports.Option = exports.Index = exports.Future = exports.Forex = exports.FOP = exports.Combo = exports.CFD = exports.Bond = exports.isNonFatalError = exports.ErrorCode = exports.IBApi = void 0;
const api_1 = require("./api/api");
// export the IB Api class and function argument types
var api_2 = require("./api/api");
Object.defineProperty(exports, "IBApi", { enumerable: true, get: function () { return api_2.IBApi; } });
var errorCode_1 = require("./common/errorCode");
Object.defineProperty(exports, "ErrorCode", { enumerable: true, get: function () { return errorCode_1.ErrorCode; } });
Object.defineProperty(exports, "isNonFatalError", { enumerable: true, get: function () { return errorCode_1.isNonFatalError; } });
// export contract types
var bond_1 = require("./api/contract/bond");
Object.defineProperty(exports, "Bond", { enumerable: true, get: function () { return bond_1.Bond; } });
var cfd_1 = require("./api/contract/cfd");
Object.defineProperty(exports, "CFD", { enumerable: true, get: function () { return cfd_1.CFD; } });
var combo_1 = require("./api/contract/combo");
Object.defineProperty(exports, "Combo", { enumerable: true, get: function () { return combo_1.Combo; } });
var fop_1 = require("./api/contract/fop");
Object.defineProperty(exports, "FOP", { enumerable: true, get: function () { return fop_1.FOP; } });
var forex_1 = require("./api/contract/forex");
Object.defineProperty(exports, "Forex", { enumerable: true, get: function () { return forex_1.Forex; } });
var future_1 = require("./api/contract/future");
Object.defineProperty(exports, "Future", { enumerable: true, get: function () { return future_1.Future; } });
var ind_1 = require("./api/contract/ind");
Object.defineProperty(exports, "Index", { enumerable: true, get: function () { return ind_1.Index; } });
var option_1 = require("./api/contract/option");
Object.defineProperty(exports, "Option", { enumerable: true, get: function () { return option_1.Option; } });
var stock_1 = require("./api/contract/stock");
Object.defineProperty(exports, "Stock", { enumerable: true, get: function () { return stock_1.Stock; } });
var wsh_1 = require("./api/contract/wsh");
Object.defineProperty(exports, "WshEventData", { enumerable: true, get: function () { return wsh_1.WshEventData; } });
// export enum types
var event_name_1 = require("./api/data/enum/event-name");
Object.defineProperty(exports, "EventName", { enumerable: true, get: function () { return event_name_1.EventName; } });
var fa_data_type_1 = require("./api/data/enum/fa-data-type");
Object.defineProperty(exports, "FADataType", { enumerable: true, get: function () { return fa_data_type_1.FADataType; } });
var log_level_1 = require("./api/data/enum/log-level");
Object.defineProperty(exports, "LogLevel", { enumerable: true, get: function () { return log_level_1.LogLevel; } });
var min_server_version_1 = require("./api/data/enum/min-server-version");
Object.defineProperty(exports, "MIN_SERVER_VER", { enumerable: true, get: function () { return min_server_version_1.MIN_SERVER_VER; } });
var option_exercise_action_1 = require("./api/data/enum/option-exercise-action");
Object.defineProperty(exports, "OptionExerciseAction", { enumerable: true, get: function () { return option_exercise_action_1.OptionExerciseAction; } });
var option_type_1 = require("./api/data/enum/option-type");
Object.defineProperty(exports, "OptionType", { enumerable: true, get: function () { return option_type_1.OptionType; } });
var sec_type_1 = require("./api/data/enum/sec-type");
Object.defineProperty(exports, "SecType", { enumerable: true, get: function () { return sec_type_1.SecType; } });
// export historic market-data types
var duration_unit_1 = require("./api/data/enum/duration-unit");
Object.defineProperty(exports, "DurationUnit", { enumerable: true, get: function () { return duration_unit_1.DurationUnit; } });
var bar_size_setting_1 = require("./api/historical/bar-size-setting");
Object.defineProperty(exports, "BarSizeSetting", { enumerable: true, get: function () { return bar_size_setting_1.BarSizeSetting; } });
__exportStar(require("./api/historical/what-to-show"), exports);
var tickByTickDataType_1 = require("./api/market/tickByTickDataType");
Object.defineProperty(exports, "TickByTickDataType", { enumerable: true, get: function () { return tickByTickDataType_1.TickByTickDataType; } });
var execution_condition_1 = require("./api/order/condition/execution-condition");
Object.defineProperty(exports, "ExecutionCondition", { enumerable: true, get: function () { return execution_condition_1.ExecutionCondition; } });
var margin_condition_1 = require("./api/order/condition/margin-condition");
Object.defineProperty(exports, "MarginCondition", { enumerable: true, get: function () { return margin_condition_1.MarginCondition; } });
var percent_change_condition_1 = require("./api/order/condition/percent-change-condition");
Object.defineProperty(exports, "PercentChangeCondition", { enumerable: true, get: function () { return percent_change_condition_1.PercentChangeCondition; } });
var price_condition_1 = require("./api/order/condition/price-condition");
Object.defineProperty(exports, "PriceCondition", { enumerable: true, get: function () { return price_condition_1.PriceCondition; } });
var time_condition_1 = require("./api/order/condition/time-condition");
Object.defineProperty(exports, "TimeCondition", { enumerable: true, get: function () { return time_condition_1.TimeCondition; } });
var volume_condition_1 = require("./api/order/condition/volume-condition");
Object.defineProperty(exports, "VolumeCondition", { enumerable: true, get: function () { return volume_condition_1.VolumeCondition; } });
// export order enum types
var conjunction_connection_1 = require("./api/order/enum/conjunction-connection");
Object.defineProperty(exports, "ConjunctionConnection", { enumerable: true, get: function () { return conjunction_connection_1.ConjunctionConnection; } });
var order_action_1 = require("./api/order/enum/order-action");
Object.defineProperty(exports, "OrderAction", { enumerable: true, get: function () { return order_action_1.OrderAction; } });
var order_condition_type_1 = require("./api/order/enum/order-condition-type");
Object.defineProperty(exports, "OrderConditionType", { enumerable: true, get: function () { return order_condition_type_1.OrderConditionType; } });
var order_status_1 = require("./api/order/enum/order-status");
Object.defineProperty(exports, "OrderStatus", { enumerable: true, get: function () { return order_status_1.OrderStatus; } });
var orderType_1 = require("./api/order/enum/orderType");
Object.defineProperty(exports, "OrderType", { enumerable: true, get: function () { return orderType_1.OrderType; } });
__exportStar(require("./api/order/enum/tif"), exports);
var trigger_method_1 = require("./api/order/enum/trigger-method");
Object.defineProperty(exports, "TriggerMethod", { enumerable: true, get: function () { return trigger_method_1.TriggerMethod; } });
var limit_1 = require("./api/order/limit");
Object.defineProperty(exports, "LimitOrder", { enumerable: true, get: function () { return limit_1.LimitOrder; } });
var liquidities_1 = require("./api/order/liquidities");
Object.defineProperty(exports, "Liquidities", { enumerable: true, get: function () { return liquidities_1.Liquidities; } });
var market_1 = require("./api/order/market");
Object.defineProperty(exports, "MarketOrder", { enumerable: true, get: function () { return market_1.MarketOrder; } });
var marketClose_1 = require("./api/order/marketClose");
Object.defineProperty(exports, "MarketCloseOrder", { enumerable: true, get: function () { return marketClose_1.MarketCloseOrder; } });
var stop_1 = require("./api/order/stop");
Object.defineProperty(exports, "StopOrder", { enumerable: true, get: function () { return stop_1.StopOrder; } });
var stopLimit_1 = require("./api/order/stopLimit");
Object.defineProperty(exports, "StopLimitOrder", { enumerable: true, get: function () { return stopLimit_1.StopLimitOrder; } });
var trailingStop_1 = require("./api/order/trailingStop");
Object.defineProperty(exports, "TrailingStopOrder", { enumerable: true, get: function () { return trailingStop_1.TrailingStopOrder; } });
// export market scanner types
var market_scanner_1 = require("./api/market-scanner/market-scanner");
Object.defineProperty(exports, "Instrument", { enumerable: true, get: function () { return market_scanner_1.Instrument; } });
Object.defineProperty(exports, "LocationCode", { enumerable: true, get: function () { return market_scanner_1.LocationCode; } });
Object.defineProperty(exports, "ScanCode", { enumerable: true, get: function () { return market_scanner_1.ScanCode; } });
// export IBApi as default
exports.default = api_1.IBApi;
// export IBApiNext types
__exportStar(require("./api-next"), exports);
//# sourceMappingURL=index.js.map