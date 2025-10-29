"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BarSizeSetting = void 0;
/**
 * Allowed bar size settings for historical market data:
 * https://interactivebrokers.github.io/tws-api/historical_bars.html
 */
var BarSizeSetting;
(function (BarSizeSetting) {
    BarSizeSetting["SECONDS_ONE"] = "1 secs";
    BarSizeSetting["SECONDS_FIVE"] = "5 secs";
    BarSizeSetting["SECONDS_TEN"] = "10 secs";
    BarSizeSetting["SECONDS_FIFTEEN"] = "15 secs";
    BarSizeSetting["SECONDS_THIRTY"] = "30 secs";
    BarSizeSetting["MINUTES_ONE"] = "1 min";
    BarSizeSetting["MINUTES_TWO"] = "2 mins";
    BarSizeSetting["MINUTES_THREE"] = "3 mins";
    BarSizeSetting["MINUTES_FIVE"] = "5 mins";
    BarSizeSetting["MINUTES_TEN"] = "10 mins";
    BarSizeSetting["MINUTES_FIFTEEN"] = "15 mins";
    BarSizeSetting["MINUTES_TWENTY"] = "20 mins";
    BarSizeSetting["MINUTES_THIRTY"] = "30 mins";
    BarSizeSetting["HOURS_ONE"] = "1 hour";
    BarSizeSetting["HOURS_TWO"] = "2 hours";
    BarSizeSetting["HOURS_THREE"] = "3 hours";
    BarSizeSetting["HOURS_FOUR"] = "4 hours";
    BarSizeSetting["HOURS_EIGHT"] = "8 hours";
    BarSizeSetting["DAYS_ONE"] = "1 day";
    BarSizeSetting["WEEKS_ONE"] = "1 week";
    BarSizeSetting["MONTHS_ONE"] = "1 month";
})(BarSizeSetting || (exports.BarSizeSetting = BarSizeSetting = {}));
exports.default = BarSizeSetting;
//# sourceMappingURL=bar-size-setting.js.map