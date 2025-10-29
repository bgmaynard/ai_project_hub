"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const event_name_1 = require("../../../api/data/enum/event-name");
const sec_type_1 = __importDefault(require("../../../api/data/enum/sec-type"));
const order_action_1 = __importDefault(require("../../../api/order/enum/order-action"));
const order_status_1 = __importDefault(require("../../../api/order/enum/order-status"));
const orderType_1 = __importDefault(require("../../../api/order/enum/orderType"));
describe("RxJS Wrapper: getAllOpenOrders", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        const openOrders = [
            {
                orderId: 0,
                contract: {
                    conId: 372615761,
                    symbol: "MOGO",
                    secType: sec_type_1.default.STK,
                    lastTradeDateOrContractMonth: "",
                    strike: 0,
                    multiplier: 0,
                    exchange: "SMART",
                    currency: "USD",
                    localSymbol: "MOGO",
                    tradingClass: "SCM",
                    comboLegsDescription: "",
                },
                order: {
                    orderId: 0,
                    action: order_action_1.default.BUY,
                    totalQuantity: 1000,
                    orderType: orderType_1.default.LMT,
                    lmtPrice: 7.1,
                    auxPrice: 0,
                    tif: __1.TimeInForce.GTC,
                    ocaGroup: "",
                    account: "DU*******",
                    openClose: "",
                    origin: 0,
                    orderRef: "",
                    clientId: 0,
                    permId: 374834303,
                    outsideRth: false,
                    hidden: false,
                    discretionaryAmt: 0,
                    goodAfterTime: "",
                    faGroup: "",
                    faMethod: "",
                    faPercentage: "",
                    modelCode: "",
                    goodTillDate: "",
                    rule80A: "0",
                    percentOffset: 1.7976931348623157e308,
                    settlingFirm: "",
                    shortSaleSlot: 0,
                    designatedLocation: "",
                    exemptCode: -1,
                    auctionStrategy: 0,
                    startingPrice: 1.7976931348623157e308,
                    stockRefPrice: 1.7976931348623157e308,
                    delta: 1.7976931348623157e308,
                    stockRangeLower: 1.7976931348623157e308,
                    stockRangeUpper: 1.7976931348623157e308,
                    displaySize: 0,
                    blockOrder: false,
                    sweepToFill: false,
                    allOrNone: false,
                    minQty: 1.7976931348623157e308,
                    ocaType: 3,
                    eTradeOnly: false,
                    firmQuoteOnly: false,
                    nbboPriceCap: 1.7976931348623157e308,
                    parentId: 0,
                    triggerMethod: 0,
                    volatility: 1.7976931348623157e308,
                    volatilityType: 0,
                    deltaNeutralOrderType: "None",
                    deltaNeutralAuxPrice: 1.7976931348623157e308,
                    deltaNeutralConId: 0,
                    deltaNeutralSettlingFirm: "",
                    deltaNeutralClearingAccount: "",
                    deltaNeutralClearingIntent: "",
                    deltaNeutralOpenClose: "?",
                    deltaNeutralShortSale: false,
                    deltaNeutralShortSaleSlot: 0,
                    deltaNeutralDesignatedLocation: "",
                    continuousUpdate: 0,
                    referencePriceType: 0,
                    trailStopPrice: 1.7976931348623157e308,
                    trailingPercent: 1.7976931348623157e308,
                    basisPoints: 1.7976931348623157e308,
                    basisPointsType: 1.7976931348623157e308,
                    scaleInitLevelSize: 0,
                    scaleSubsLevelSize: 1.7976931348623157e308,
                    scalePriceIncrement: 1.7976931348623157e308,
                    hedgeType: "",
                    optOutSmartRouting: false,
                    clearingAccount: "0",
                    clearingIntent: "",
                    notHeld: false,
                    algoStrategy: "0",
                    solicited: false,
                    whatIf: false,
                    randomizeSize: false,
                    randomizePrice: false,
                    conditions: [],
                    adjustedOrderType: "None",
                    triggerPrice: 1.7976931348623157e308,
                    lmtPriceOffset: 1.7976931348623157e308,
                    adjustedStopPrice: 1.7976931348623157e308,
                    adjustedStopLimitPrice: 1.7976931348623157e308,
                    adjustedTrailingAmount: 1.7976931348623157e308,
                    adjustableTrailingUnit: 0,
                    softDollarTier: {
                        name: "",
                        value: "",
                        displayName: "",
                    },
                    cashQty: 0,
                    dontUseAutoPriceForHedge: true,
                    isOmsContainer: false,
                    discretionaryUpToLimitPrice: false,
                    usePriceMgmtAlgo: true,
                },
                orderState: {
                    status: order_status_1.default.Submitted,
                    commissionCurrency: "",
                    warningText: "",
                },
            },
        ];
        apiNext.getAllOpenOrders().then((data) => {
            expect(data.length).toEqual(1);
            expect(data[0]["orderId"]).toMatchObject(openOrders);
            done();
        });
        api.emit(event_name_1.EventName.openOrder, openOrders);
        api.emit(event_name_1.EventName.openOrderEnd);
    });
});
//# sourceMappingURL=get-all-open-orders.test.js.map