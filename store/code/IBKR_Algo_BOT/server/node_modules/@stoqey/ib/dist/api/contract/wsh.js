"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WshEventData = void 0;
/**
 * A WshEventData event.
 */
class WshEventData {
    constructor(conId, fillWatchlist = false, fillPortfolio = false, fillCompetitors = false, startDate = "", endDate = "", totalLimit = 0) {
        this.conId = conId;
        this.fillWatchlist = fillWatchlist;
        this.fillPortfolio = fillPortfolio;
        this.fillCompetitors = fillCompetitors;
        this.startDate = startDate;
        this.endDate = endDate;
        this.totalLimit = totalLimit;
        this.filter = "";
    }
}
exports.WshEventData = WshEventData;
exports.default = WshEventData;
//# sourceMappingURL=wsh.js.map