"use strict";
/**
 * [[IBApi]] error event codes.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.isNonFatalError = exports.ErrorCode = void 0;
/* eslint-disable @typescript-eslint/no-duplicate-enum-values */
var ErrorCode;
(function (ErrorCode) {
    /** No request id associated to this error */
    ErrorCode[ErrorCode["NO_VALID_ID"] = -1] = "NO_VALID_ID";
    /** Order Canceled - reason: */
    ErrorCode[ErrorCode["ORDER_CANCELLED"] = 202] = "ORDER_CANCELLED";
    /** Order Message: BUY 1 SPY ARCA Warning: your order will not be placed at the exchange until 2024-12-02 04:00:00 US/Eastern (#399) */
    ErrorCode[ErrorCode["ORDER_MESSAGE"] = 399] = "ORDER_MESSAGE";
    /** You must subscribe for additional permissions to obtain precise results for scanner.Parameter:Most Active,Filter:Price;Real-Time Market Data:Pink Sheets */
    ErrorCode[ErrorCode["SCANNER_LOW_PRECISION"] = 492] = "SCANNER_LOW_PRECISION";
    /** Requested market data is not subscribed. Delayed market data is not available. */
    ErrorCode[ErrorCode["REQ_MKT_DATA_NOT_AVAIL"] = 354] = "REQ_MKT_DATA_NOT_AVAIL";
    /** No trading permissions. */
    ErrorCode[ErrorCode["NO_TRADING_PERMISSIONS"] = 460] = "NO_TRADING_PERMISSIONS";
    /** Already connected. */
    ErrorCode[ErrorCode["ALREADY_CONNECTED"] = 501] = "ALREADY_CONNECTED";
    /**
     * Couldn't connect to TWS.
     *
     * Confirm that "Enable ActiveX and Socket Clients" is enabled and connection port is the same as Socket Port on
     * the TWS  "Edit->Global Configuration...->API->Settings" menu.
     *
     * Live Trading ports: TWS: 7496; IB Gateway: 4001.
     *
     * Simulated Trading ports for new installations of version 954.1 or newer: TWS: 7497; IB Gateway: 4002
     */
    ErrorCode[ErrorCode["CONNECT_FAIL"] = 502] = "CONNECT_FAIL";
    /** The TWS is out of date and must be upgraded. */
    ErrorCode[ErrorCode["UPDATE_TWS"] = 503] = "UPDATE_TWS";
    /** Not connected. */
    ErrorCode[ErrorCode["NOT_CONNECTED"] = 504] = "NOT_CONNECTED";
    /** Fatal Error: Unknown message id. */
    ErrorCode[ErrorCode["UNKNOWN_ID"] = 505] = "UNKNOWN_ID";
    /** Unsupported Version. */
    ErrorCode[ErrorCode["UNSUPPORTED_VERSION"] = 506] = "UNSUPPORTED_VERSION";
    /** Bad Message Length. */
    ErrorCode[ErrorCode["BAD_LENGTH"] = 507] = "BAD_LENGTH";
    /** Bad Message. */
    ErrorCode[ErrorCode["BAD_MESSAGE"] = 508] = "BAD_MESSAGE";
    /** Failed to send message. */
    ErrorCode[ErrorCode["FAIL_SEND"] = 509] = "FAIL_SEND";
    /** Request Market Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMKT"] = 510] = "FAIL_SEND_REQMKT";
    /** Cancel Market Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANMKT"] = 511] = "FAIL_SEND_CANMKT";
    /** Order Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_ORDER"] = 512] = "FAIL_SEND_ORDER";
    /** Account Update Request Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_ACCT"] = 513] = "FAIL_SEND_ACCT";
    /** Request For Executions Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_EXEC"] = 514] = "FAIL_SEND_EXEC";
    /** Cancel Order Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CORDER"] = 515] = "FAIL_SEND_CORDER";
    /** Request Open Order Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_OORDER"] = 516] = "FAIL_SEND_OORDER";
    /** Unknown contract. Verify the contract details supplied. */
    ErrorCode[ErrorCode["UNKNOWN_CONTRACT"] = 517] = "UNKNOWN_CONTRACT";
    /** Request Contract Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQCONTRACT"] = 518] = "FAIL_SEND_REQCONTRACT";
    /** Request Market Depth Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMKTDEPTH"] = 519] = "FAIL_SEND_REQMKTDEPTH";
    /** Cancel Market Depth Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANMKTDEPTH"] = 520] = "FAIL_SEND_CANMKTDEPTH";
    /** Set Server Log Level Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_SERVER_LOG_LEVEL"] = 521] = "FAIL_SEND_SERVER_LOG_LEVEL";
    /** FA Information Request Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_FA_REQUEST"] = 522] = "FAIL_SEND_FA_REQUEST";
    /** FA Information Replace Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_FA_REPLACE"] = 523] = "FAIL_SEND_FA_REPLACE";
    /** Request Scanner Subscription Sending Error . */
    ErrorCode[ErrorCode["FAIL_SEND_REQSCANNER"] = 524] = "FAIL_SEND_REQSCANNER";
    /** Cancel Scanner Subscription Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANSCANNER"] = 525] = "FAIL_SEND_CANSCANNER";
    /** Request Scanner Parameter Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQSCANNERPARAMETERS"] = 526] = "FAIL_SEND_REQSCANNERPARAMETERS";
    /** Request Historical Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQHISTDATA"] = 527] = "FAIL_SEND_REQHISTDATA";
    /** Request Historical Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANHISTDATA"] = 528] = "FAIL_SEND_CANHISTDATA";
    /** Request Real-time Bar Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQRTBARS"] = 529] = "FAIL_SEND_REQRTBARS";
    /** Cancel Real-time Bar Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANRTBARS"] = 530] = "FAIL_SEND_CANRTBARS";
    /** Request Current Time Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQCURRTIME"] = 531] = "FAIL_SEND_REQCURRTIME";
    /** Request Fundamental Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQFUNDDATA"] = 532] = "FAIL_SEND_REQFUNDDATA";
    /** Cancel Fundamental Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANFUNDDATA"] = 533] = "FAIL_SEND_CANFUNDDATA";
    /** Request Calculate Implied Volatility Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQCALCIMPLIEDVOLAT"] = 534] = "FAIL_SEND_REQCALCIMPLIEDVOLAT";
    /** Request Calculate Option Price Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQCALCOPTIONPRICE"] = 535] = "FAIL_SEND_REQCALCOPTIONPRICE";
    /** Cancel Calculate Implied Volatility Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANCALCIMPLIEDVOLAT"] = 536] = "FAIL_SEND_CANCALCIMPLIEDVOLAT";
    /** Cancel Calculate Option Price Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANCALCOPTIONPRICE"] = 537] = "FAIL_SEND_CANCALCOPTIONPRICE";
    /** Request Global Cancel Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQGLOBALCANCEL"] = 538] = "FAIL_SEND_REQGLOBALCANCEL";
    /** Request Market Data Type Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMARKETDATATYPE"] = 539] = "FAIL_SEND_REQMARKETDATATYPE";
    /** Request Positions Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQPOSITIONS"] = 540] = "FAIL_SEND_REQPOSITIONS";
    /** Cancel Positions Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANPOSITIONS"] = 541] = "FAIL_SEND_CANPOSITIONS";
    /** Request Account Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQACCOUNTDATA"] = 542] = "FAIL_SEND_REQACCOUNTDATA";
    /** Cancel Account Data Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANACCOUNTDATA"] = 543] = "FAIL_SEND_CANACCOUNTDATA";
    /** Verify Request Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_VERIFYREQUEST"] = 544] = "FAIL_SEND_VERIFYREQUEST";
    /** Verify Message Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_VERIFYMESSAGE"] = 545] = "FAIL_SEND_VERIFYMESSAGE";
    /** Query Display Groups Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_QUERYDISPLAYGROUPS"] = 546] = "FAIL_SEND_QUERYDISPLAYGROUPS";
    /** Subscribe To Group Events Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_SUBSCRIBETOGROUPEVENTS"] = 547] = "FAIL_SEND_SUBSCRIBETOGROUPEVENTS";
    /** Update Display Group Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_UPDATEDISPLAYGROUP"] = 548] = "FAIL_SEND_UPDATEDISPLAYGROUP";
    /** Unsubscribe From Group Events Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_UNSUBSCRIBEFROMGROUPEVENTS"] = 549] = "FAIL_SEND_UNSUBSCRIBEFROMGROUPEVENTS";
    /** Start API Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_STARTAPI"] = 550] = "FAIL_SEND_STARTAPI";
    /** Verify And Auth Request Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_VERIFYANDAUTHREQUEST"] = 551] = "FAIL_SEND_VERIFYANDAUTHREQUEST";
    /** Verify And Auth Message Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_VERIFYANDAUTHMESSAGE"] = 552] = "FAIL_SEND_VERIFYANDAUTHMESSAGE";
    /** Request Positions Multi Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQPOSITIONSMULTI"] = 553] = "FAIL_SEND_REQPOSITIONSMULTI";
    /** Cancel Positions Multi Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANPOSITIONSMULTI"] = 554] = "FAIL_SEND_CANPOSITIONSMULTI";
    /** Request Account Updates Multi Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQACCOUNTUPDATESMULTI"] = 555] = "FAIL_SEND_REQACCOUNTUPDATESMULTI";
    /** Cancel Account Updates Multi Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANACCOUNTUPDATESMULTI"] = 556] = "FAIL_SEND_CANACCOUNTUPDATESMULTI";
    /** Request Security Definition Option Params Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQSECDEFOPTPARAMS"] = 557] = "FAIL_SEND_REQSECDEFOPTPARAMS";
    /** Request Soft Dollar Tiers Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQSOFTDOLLARTIERS"] = 558] = "FAIL_SEND_REQSOFTDOLLARTIERS";
    /** Request Family Codes Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQFAMILYCODES"] = 559] = "FAIL_SEND_REQFAMILYCODES";
    /** Request Matching Symbols Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMATCHINGSYMBOLS"] = 560] = "FAIL_SEND_REQMATCHINGSYMBOLS";
    /** Request Market Depth Exchanges Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMKTDEPTHEXCHANGES"] = 561] = "FAIL_SEND_REQMKTDEPTHEXCHANGES";
    /** Request Smart Components Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQSMARTCOMPONENTS"] = 562] = "FAIL_SEND_REQSMARTCOMPONENTS";
    /** Request News Providers Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQNEWSPROVIDERS"] = 563] = "FAIL_SEND_REQNEWSPROVIDERS";
    /** Request News Article Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQNEWSARTICLE"] = 564] = "FAIL_SEND_REQNEWSARTICLE";
    /** Request Historical News Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQHISTORICALNEWS"] = 565] = "FAIL_SEND_REQHISTORICALNEWS";
    /** Request Head Time Stamp Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQHEADTIMESTAMP"] = 566] = "FAIL_SEND_REQHEADTIMESTAMP";
    /** CRequest Histogram Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_REQHISTOGRAMDATA"] = 567] = "FAIL_SEND_REQHISTOGRAMDATA";
    /** Request Market Rule Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANCELHISTOGRAMDATA"] = 568] = "FAIL_SEND_CANCELHISTOGRAMDATA";
    /** Cancel PnL Single Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANCELHEADTIMESTAMP"] = 569] = "FAIL_SEND_CANCELHEADTIMESTAMP";
    /** Request Historical Ticks Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_HISTORICAL_TICK"] = 569] = "FAIL_SEND_HISTORICAL_TICK";
    /** Request Market Rule Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQMARKETRULE"] = 570] = "FAIL_SEND_REQMARKETRULE";
    /** Request PnL Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQPNL"] = 571] = "FAIL_SEND_REQPNL";
    /** Cancel PnL Sending Error. */
    ErrorCode[ErrorCode["FAIL_SEND_CANCELPNL"] = 572] = "FAIL_SEND_CANCELPNL";
    /** Request PnL Single Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQPNLSINGLE"] = 573] = "FAIL_SEND_REQPNLSINGLE";
    /** Cancel PnL Single Sending Error . */
    ErrorCode[ErrorCode["FAIL_SEND_CANCELPNLSINGLE"] = 574] = "FAIL_SEND_CANCELPNLSINGLE";
    /** Request Historical Ticks Error. */
    ErrorCode[ErrorCode["FAIL_SEND_REQHISTORICALTICKS"] = 575] = "FAIL_SEND_REQHISTORICALTICKS";
    /** Request Tick-By-Tick Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_REQTICKBYTICKDATA"] = 576] = "FAIL_SEND_REQTICKBYTICKDATA";
    /** Cancel Tick-By-Tick Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_CANCELTICKBYTICKDATA"] = 577] = "FAIL_SEND_CANCELTICKBYTICKDATA";
    /** Request Completed Orders Sending Error  */
    ErrorCode[ErrorCode["FAIL_SEND_REQCOMPLETEDORDERS"] = 578] = "FAIL_SEND_REQCOMPLETEDORDERS";
    /** Invalid symbol in string */
    ErrorCode[ErrorCode["INVALID_SYMBOL"] = 579] = "INVALID_SYMBOL";
    /** "Request WSH Meta Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_REQ_WSH_META_DATA"] = 580] = "FAIL_SEND_REQ_WSH_META_DATA";
    /** Cancel WSH Meta Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_CAN_WSH_META_DATA"] = 581] = "FAIL_SEND_CAN_WSH_META_DATA";
    /** Request WSH Event Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_REQ_WSH_EVENT_DATA"] = 582] = "FAIL_SEND_REQ_WSH_EVENT_DATA";
    /** Cancel WSH Event Data Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_CAN_WSH_EVENT_DATA"] = 583] = "FAIL_SEND_CAN_WSH_EVENT_DATA";
    /** Request User Info Sending Error */
    ErrorCode[ErrorCode["FAIL_SEND_REQ_USER_INFO"] = 584] = "FAIL_SEND_REQ_USER_INFO";
    /** FA Profile is not supported anymore, use FA Group instead */
    ErrorCode[ErrorCode["FA_PROFILE_NOT_SUPPORTED"] = 585] = "FA_PROFILE_NOT_SUPPORTED";
    /** Failed to read message because not connected */
    ErrorCode[ErrorCode["FAIL_READ_MESSAGE"] = 586] = "FAIL_READ_MESSAGE";
    /** Connectivity between IBKR and Trader Workstation has been lost.
     * @When client is connected to TWS and server but TWS losses connection to server.
     * 🗣️🗣️🗣️ "Market data connection lost".
     */
    ErrorCode[ErrorCode["FAIL_CONNECTION_LOST_BETWEEN_SERVER_AND_TWS"] = 1100] = "FAIL_CONNECTION_LOST_BETWEEN_SERVER_AND_TWS";
    /** Connectivity between Trader Workstation and server is broken. It will be restored automatically.
     * @When client connects to TWS but TWS isn't connected to server.
     */
    ErrorCode[ErrorCode["FAIL_CONNECTION_LOST_BETWEEN_TWS_AND_SERVER"] = 2110] = "FAIL_CONNECTION_LOST_BETWEEN_TWS_AND_SERVER";
    /** Invalid position trade derived value */
    ErrorCode[ErrorCode["INVALID_POSITION_TRADE_DERIVATED_VALUE"] = 2150] = "INVALID_POSITION_TRADE_DERIVATED_VALUE";
    /** Part of requested market data is not subscribed. */
    ErrorCode[ErrorCode["PART_OF_REQUESTED_DATA_NOT_SUBSCRIBED"] = 10090] = "PART_OF_REQUESTED_DATA_NOT_SUBSCRIBED";
    /** Requested market data is not subscribed. Displaying delayed market data. */
    ErrorCode[ErrorCode["DISPLAYING_DELAYED_DATA"] = 10167] = "DISPLAYING_DELAYED_DATA";
    /* News feed is not allowed. */
    ErrorCode[ErrorCode["NEWS_FEED_NOT_ALLOWED"] = 10276] = "NEWS_FEED_NOT_ALLOWED";
})(ErrorCode || (exports.ErrorCode = ErrorCode = {}));
/**
 * Check if an error message is fatal (to the request execution) or only a warning/information message
 */
const isNonFatalError = (code, error) => {
    if (code >= 2100 && code < 3000)
        return true;
    if (error.message.includes("Warning:"))
        return true;
    switch (code) {
        case ErrorCode.PART_OF_REQUESTED_DATA_NOT_SUBSCRIBED:
        case ErrorCode.DISPLAYING_DELAYED_DATA:
        case ErrorCode.ORDER_MESSAGE:
        case ErrorCode.SCANNER_LOW_PRECISION:
            return true;
        default:
            return false;
    }
};
exports.isNonFatalError = isNonFatalError;
//# sourceMappingURL=errorCode.js.map