"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isPegMidOrder = exports.isPegBestOrder = exports.isPegBenchOrder = exports.isVolOrder = exports.OrderType = void 0;
/**
 * Order types.
 */
var OrderType;
(function (OrderType) {
    OrderType["None"] = "";
    OrderType["MKT"] = "MKT";
    OrderType["LMT"] = "LMT";
    OrderType["STP"] = "STP";
    OrderType["STP_LMT"] = "STP LMT";
    OrderType["REL"] = "REL";
    OrderType["TRAIL"] = "TRAIL";
    OrderType["BOX_TOP"] = "BOX TOP";
    OrderType["FIX_PEGGED"] = "FIX PEGGED";
    OrderType["LIT"] = "LIT";
    OrderType["LMT_PLUS_MKT"] = "LMT + MKT";
    OrderType["LOC"] = "LOC";
    OrderType["MIDPRICE"] = "MIDPRICE";
    OrderType["MIT"] = "MIT";
    OrderType["MKT_PRT"] = "MKT PRT";
    OrderType["MOC"] = "MOC";
    OrderType["MTL"] = "MTL";
    OrderType["PASSV_REL"] = "PASSV REL";
    OrderType["PEG_BENCH"] = "PEG BENCH";
    OrderType["PEG_BEST"] = "PEG BEST";
    OrderType["PEG_MID"] = "PEG MID";
    OrderType["PEG_MKT"] = "PEG MKT";
    OrderType["PEG_PRIM"] = "PEG PRIM";
    OrderType["PEG_STK"] = "PEG STK";
    OrderType["REL_PLUS_LMT"] = "REL + LMT";
    OrderType["REL_PLUS_MKT"] = "REL + MKT";
    OrderType["SNAP_MID"] = "SNAP MID";
    OrderType["SNAP_MKT"] = "SNAP MKT";
    OrderType["SNAP_PRIM"] = "SNAP PRIM";
    OrderType["STP_PRT"] = "STP PRT";
    OrderType["TRAIL_LIMIT"] = "TRAIL LIMIT";
    OrderType["TRAIL_LIT"] = "TRAIL LIT";
    OrderType["TRAIL_LMT_PLUS_MKT"] = "TRAIL LMT + MKT";
    OrderType["TRAIL_MIT"] = "TRAIL MIT";
    OrderType["TRAIL_REL_PLUS_MKT"] = "TRAIL REL + MKT";
    OrderType["VOL"] = "VOL";
    OrderType["VWAP"] = "VWAP";
    OrderType["QUOTE"] = "QUOTE";
    OrderType["PEG_PRIM_VOL"] = "PPV";
    OrderType["PEG_MID_VOL"] = "PDV";
    OrderType["PEG_MKT_VOL"] = "PMV";
    OrderType["PEG_SRF_VOL"] = "PSV";
})(OrderType || (exports.OrderType = OrderType = {}));
const isVolOrder = (orderType) => orderType == OrderType.VOL;
exports.isVolOrder = isVolOrder;
const isPegBenchOrder = (orderType) => {
    if (orderType == OrderType.PEG_BENCH || orderType == "PEGBENCH")
        return true;
    else
        return false;
};
exports.isPegBenchOrder = isPegBenchOrder;
const isPegBestOrder = (orderType) => {
    if (orderType == OrderType.PEG_BEST || orderType == "PEGBEST")
        return true;
    else
        return false;
};
exports.isPegBestOrder = isPegBestOrder;
const isPegMidOrder = (orderType) => {
    if (orderType == OrderType.PEG_MID || orderType == "PEGMID")
        return true;
    else
        return false;
};
exports.isPegMidOrder = isPegMidOrder;
exports.default = OrderType;
//# sourceMappingURL=orderType.js.map