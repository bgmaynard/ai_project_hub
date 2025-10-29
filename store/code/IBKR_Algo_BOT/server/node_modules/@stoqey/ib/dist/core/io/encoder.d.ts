import { MarketDataType, OrderCancel, WhatToShow } from "../..";
import { Contract } from "../../api/contract/contract";
import WshEventData from "../../api/contract/wsh";
import TagValue from "../../api/data/container/tag-value";
import FADataType from "../../api/data/enum/fa-data-type";
import LogLevel from "../../api/data/enum/log-level";
import OptionExerciseAction from "../../api/data/enum/option-exercise-action";
import { BarSizeSetting } from "../../api/historical/bar-size-setting";
import { ScannerSubscription } from "../../api/market/scannerSubscription";
import { TickByTickDataType } from "../../api/market/tickByTickDataType";
import { Order } from "../../api/order/order";
import { ExecutionFilter } from "../../api/report/executionFilter";
/**
 * @internal
 *
 * Outgoing message IDs.
 */
export declare enum OUT_MSG_ID {
    REQ_MKT_DATA = 1,
    CANCEL_MKT_DATA = 2,
    PLACE_ORDER = 3,
    CANCEL_ORDER = 4,
    REQ_OPEN_ORDERS = 5,
    REQ_ACCOUNT_DATA = 6,
    REQ_EXECUTIONS = 7,
    REQ_IDS = 8,
    REQ_CONTRACT_DATA = 9,
    REQ_MKT_DEPTH = 10,
    CANCEL_MKT_DEPTH = 11,
    REQ_NEWS_BULLETINS = 12,
    CANCEL_NEWS_BULLETINS = 13,
    SET_SERVER_LOGLEVEL = 14,
    REQ_AUTO_OPEN_ORDERS = 15,
    REQ_ALL_OPEN_ORDERS = 16,
    REQ_MANAGED_ACCTS = 17,
    REQ_FA = 18,
    REPLACE_FA = 19,
    REQ_HISTORICAL_DATA = 20,
    EXERCISE_OPTIONS = 21,
    REQ_SCANNER_SUBSCRIPTION = 22,
    CANCEL_SCANNER_SUBSCRIPTION = 23,
    REQ_SCANNER_PARAMETERS = 24,
    CANCEL_HISTORICAL_DATA = 25,
    REQ_CURRENT_TIME = 49,
    REQ_REAL_TIME_BARS = 50,
    CANCEL_REAL_TIME_BARS = 51,
    REQ_FUNDAMENTAL_DATA = 52,
    CANCEL_FUNDAMENTAL_DATA = 53,
    REQ_CALC_IMPLIED_VOLAT = 54,
    REQ_CALC_OPTION_PRICE = 55,
    CANCEL_CALC_IMPLIED_VOLAT = 56,
    CANCEL_CALC_OPTION_PRICE = 57,
    REQ_GLOBAL_CANCEL = 58,
    REQ_MARKET_DATA_TYPE = 59,
    REQ_POSITIONS = 61,
    REQ_ACCOUNT_SUMMARY = 62,
    CANCEL_ACCOUNT_SUMMARY = 63,
    CANCEL_POSITIONS = 64,
    VERIFY_REQUEST = 65,// not implemented
    VERIFY_MESSAGE = 66,// not implemented
    QUERY_DISPLAY_GROUPS = 67,
    SUBSCRIBE_TO_GROUP_EVENTS = 68,
    UPDATE_DISPLAY_GROUP = 69,
    UNSUBSCRIBE_FROM_GROUP_EVENTS = 70,
    START_API = 71,// sent by [[Socket]]
    VERIFY_AND_AUTH_REQUEST = 72,// not implemented
    VERIFY_AND_AUTH_MESSAGE = 73,// not implemented
    REQ_POSITIONS_MULTI = 74,
    CANCEL_POSITIONS_MULTI = 75,
    REQ_ACCOUNT_UPDATES_MULTI = 76,
    CANCEL_ACCOUNT_UPDATES_MULTI = 77,
    REQ_SEC_DEF_OPT_PARAMS = 78,
    REQ_SOFT_DOLLAR_TIERS = 79,
    REQ_FAMILY_CODES = 80,
    REQ_MATCHING_SYMBOLS = 81,
    REQ_MKT_DEPTH_EXCHANGES = 82,
    REQ_SMART_COMPONENTS = 83,
    REQ_NEWS_ARTICLE = 84,
    REQ_NEWS_PROVIDERS = 85,
    REQ_HISTORICAL_NEWS = 86,
    REQ_HEAD_TIMESTAMP = 87,
    REQ_HISTOGRAM_DATA = 88,
    CANCEL_HISTOGRAM_DATA = 89,
    CANCEL_HEAD_TIMESTAMP = 90,
    REQ_MARKET_RULE = 91,
    REQ_PNL = 92,
    CANCEL_PNL = 93,
    REQ_PNL_SINGLE = 94,
    CANCEL_PNL_SINGLE = 95,
    REQ_HISTORICAL_TICKS = 96,
    REQ_TICK_BY_TICK_DATA = 97,
    CANCEL_TICK_BY_TICK_DATA = 98,
    REQ_COMPLETED_ORDERS = 99,
    REQ_WSH_META_DATA = 100,
    CANCEL_WSH_META_DATA = 101,
    REQ_WSH_EVENT_DATA = 102,
    CANCEL_WSH_EVENT_DATA = 103,
    REQ_USER_INFO = 104
}
/**
 * @internal
 *
 * Callback interface of the [[Encoder]].
 */
export interface EncoderCallbacks {
    /** Get the IB API server version. */
    readonly serverVersion: number;
    /**
     * Send a message to the server connection.
     *
     * @param args Array of tokens to send.
     * Can contain nested arrays.
     */
    sendMsg(...tokens: unknown[]): void;
    /**
     * Emit an error event to public API interface.
     *
     * @param errMsg The error test message.
     * @param data Additional error data (optional).
     */
    emitError(errMsg: string, code: number, reqId: number): void;
}
/**
 * @internal
 *
 * Class for encoding messages and sending raw token data back to
 */
export declare class Encoder {
    private callback;
    /**
     * Create an [[Encoder]] object for encoding messages to token data.
     *
     * @param callback A [[EncoderCallbacks]] implementation.
     */
    constructor(callback: EncoderCallbacks);
    /** Get the API server version. */
    private get serverVersion();
    /**
     * Send a message to the server connection.
     *
     * @param args Array of tokens to send.
     */
    private sendMsg;
    /**
     * Emit an error event to public API interface.
     *
     * @param errMsg The error test message.
     * @param data Additional error data (optional).
     */
    private emitError;
    /**
     * Encode a [[Contract]] to an array of tokens.
     */
    private encodeContract;
    /**
     * Encode a [[TagValue]] array to a string token.
     */
    private encodeTagValues;
    /**
     * @@internal
     *
     * Helper function convert an array of [[TagValue]] to a flat [tag,value] tuple array.
     */
    /**
     * Encode a calculateImpliedVolatility message to an array of tokens.
     */
    calculateImpliedVolatility(reqId: number, contract: Contract, optionPrice: number, underPrice: number, impliedVolatilityOptions?: TagValue[]): void;
    /**
     * Encode a calculateOptionPrice message to an array of tokens.
     */
    calculateOptionPrice(reqId: number, contract: Contract, volatility: number, underPrice: number, optionPriceOptions?: TagValue[]): void;
    /**
     * Encode a CANCEL_ACCOUNT_SUMMARY message to an array of tokens.
     */
    cancelAccountSummary(reqId: number): void;
    /**
     * Encode a CANCEL_ACCOUNT_UPDATES_MULTI message to an array of tokens.
     */
    cancelAccountUpdatesMulti(reqId: number): void;
    /**
     * Encode a CANCEL_CALC_IMPLIED_VOLAT message to an array of tokens.
     */
    cancelCalculateImpliedVolatility(reqId: number): void;
    /**
     * Encode a CANCEL_CALC_OPTION_PRICE message to an array of tokens.
     */
    cancelCalculateOptionPrice(reqId: number): void;
    /**
     * Encode a CANCEL_FUNDAMENTAL_DATA message to an array of tokens.
     */
    cancelFundamentalData(reqId: number): void;
    /**
     * Encode a CANCEL_HISTORICAL_DATA message to an array of tokens.
     */
    cancelHistoricalData(reqId: number): void;
    /**
     * Encode a CANCEL_MKT_DATA message to an array of tokens.
     */
    cancelMktData(reqId: number): void;
    /**
     * Encode a CANCEL_MKT_DEPTH message to an array of tokens.
     */
    cancelMktDepth(reqId: number, isSmartDepth: boolean): void;
    /**
     * Encode a CANCEL_NEWS_BULLETINS message to an array of tokens.
     */
    cancelNewsBulletins(): void;
    /**
     * Encode a CANCEL_ORDER message to an array of tokens.
     */
    cancelOrder(orderId: number, orderCancel: OrderCancel): void;
    /**
     * Encode a CANCEL_POSITIONS message to an array of tokens.
     */
    cancelPositions(): void;
    /**
     * Encode a CANCEL_REAL_TIME_BARS message to an array of tokens.
     */
    cancelRealTimeBars(reqId: number): void;
    /**
     * Encode a CANCEL_SCANNER_SUBSCRIPTION message to an array of tokens.
     */
    cancelScannerSubscription(reqId: number): void;
    /**
     * Encode a EXERCISE_OPTIONS message to an array of tokens.
     */
    exerciseOptions(reqId: number, contract: Contract, exerciseAction: OptionExerciseAction, exerciseQuantity: number, account: string, override: number, manualOrderTime?: string, customerAccount?: string, professionalCustomer?: boolean): void;
    /**
     * Encode a PLACE_ORDER message to an array of tokens.
     */
    placeOrder(id: number, contract: Contract, order: Order): void;
    /**
     * Encode a REPLACE_FA message to an array of tokens.
     */
    replaceFA(reqId: number, faDataType: FADataType, xml: string): void;
    /**
     * Encode a REQ_ACCOUNT_SUMMARY message to an array of tokens.
     */
    reqAccountSummary(reqId: number, group: string, tags: string): void;
    /**
     * Encode a REQ_PNL message to an array of tokens.
     */
    reqPnL(reqId: number, account: string, modelCode: string | null): void;
    /**
     * Encode a REQ_PNL message.
     */
    cancelPnL(reqId: number): void;
    /**
     * Encode a REQ_PNL_SINGLE message.
     */
    reqPnLSingle(reqId: number, account: string, modelCode: string | null, conId: number): void;
    /**
     * Encode a CANCEL_PNL_SINGLE message.
     */
    cancelPnLSingle(reqId: number): void;
    /**
     * Encode a REQ_ACCOUNT_DATA message.
     */
    reqAccountUpdates(subscribe: boolean, acctCode: string): void;
    /**
     * Encode a REQ_ACCOUNT_UPDATES_MULTI message.
     */
    reqAccountUpdatesMulti(reqId: number, acctCode: string, modelCode: string, ledgerAndNLV: boolean): void;
    /**
     * Encode a REQ_ALL_OPEN_ORDERS message.
     */
    reqAllOpenOrders(): void;
    /**
     * Encode a REQ_AUTO_OPEN_ORDERS message.
     */
    reqAutoOpenOrders(bAutoBind: boolean): void;
    /**
     * Encode a REQ_HEAD_TIMESTAMP message.
     */
    reqHeadTimestamp(reqId: number, contract: Contract, whatToShow: WhatToShow, useRTH: boolean, formatDate: number): void;
    /**
     * Encode a REQ_CONTRACT_DATA message.
     */
    reqContractDetails(reqId: number, contract: Contract): void;
    /**
     * Encode a REQ_CURRENT_TIME message.
     */
    reqCurrentTime(): void;
    /**
     * Encode a REQ_EXECUTIONS message.
     */
    reqExecutions(reqId: number, filter: ExecutionFilter): void;
    /**
     * Encode a REQ_FUNDAMENTAL_DATA message.
     */
    reqFundamentalData(reqId: number, contract: Contract, reportType: string, fundamentalDataOptions: TagValue[]): void;
    /**
     * Encode a REQ_GLOBAL_CANCEL message.
     */
    reqGlobalCancel(orderCancel: OrderCancel): void;
    /**
     * Encode a REQ_HISTORICAL_DATA message.
     */
    reqHistoricalData(reqId: number, contract: Contract, endDateTime: string, durationStr: string, barSizeSetting: BarSizeSetting, whatToShow: WhatToShow, useRTH: number | boolean, formatDate: number, keepUpToDate: boolean, chartOptions?: TagValue[]): void;
    /**
     * Encode a REQ_HISTORICAL_TICKS message.
     */
    reqHistoricalTicks(reqId: number, contract: Contract, startDateTime: string, endDateTime: string, numberOfTicks: number, whatToShow: WhatToShow, useRth: number | boolean, ignoreSize: boolean, miscOptions?: TagValue[]): void;
    /**
     * Encode a REQ_TICK_BY_TICK_DATA message.
     */
    encodeReqTickByTickData(reqId: number, contract: Contract, tickType: TickByTickDataType, numberOfTicks: number, ignoreSize: boolean): unknown[];
    reqTickByTickData(reqId: number, contract: Contract, tickType: TickByTickDataType, numberOfTicks: number, ignoreSize: boolean): void;
    /**
     * Encode a CANCEL_TICK_BY_TICK_DATA message.
     */
    cancelTickByTickData(reqId: number): void;
    /**
     * Encode a REQ_IDS message.
     */
    reqIds(numIds: number): void;
    /**
     * Encode a REQ_IDS message.
     */
    reqManagedAccts(): void;
    /**
     * Encode a REQ_MARKET_DATA_TYPE message.
     */
    reqMarketDataType(marketDataType: MarketDataType): void;
    /**
     * Encode a REQ_MKT_DATA message.
     */
    reqMktData(reqId: number, contract: Contract, genericTickList: string, snapshot: boolean, regulatorySnapshot: boolean): void;
    /**
     * Encode a REQ_MKT_DEPTH message.
     */
    reqMktDepth(reqId: number, contract: Contract, numRows: number, isSmartDepth: boolean, mktDepthOptions?: TagValue[]): void;
    /**
     * Encode a REQ_NEWS_BULLETINS message.
     */
    reqNewsBulletins(allMsgs: boolean): void;
    /**
     * Encode a REQ_OPEN_ORDERS message.
     */
    reqOpenOrders(): void;
    /**
     * Encode a REQ_POSITIONS message.
     */
    reqPositions(): void;
    /**
     * Encode a REQ_POSITIONS_MULTI message.
     */
    reqPositionsMulti(reqId: number, account: string, modelCode: string | null): void;
    /**
     * Encode a CANCEL_POSITIONS_MULTI message.
     */
    cancelPositionsMulti(reqId: number): void;
    /**
     * Encode a REQ_REAL_TIME_BARS message.
     */
    reqRealTimeBars(reqId: number, contract: Contract, barSize: number, whatToShow: WhatToShow, useRTH: boolean, realTimeBarsOptions: TagValue[]): void;
    /**
     * Encode a REQ_SCANNER_PARAMETERS message.
     */
    reqScannerParameters(): void;
    /**
     * Encode a REQ_SCANNER_SUBSCRIPTION message.
     */
    reqScannerSubscription(reqId: number, subscription: ScannerSubscription, scannerSubscriptionOptions: TagValue[], scannerSubscriptionFilterOptions?: TagValue[]): void;
    /**
     * Encode a REQ_FA message.
     */
    requestFA(faDataType: FADataType): void;
    /**
     * Encode a SET_SERVER_LOGLEVEL message.
     */
    setServerLogLevel(logLevel: LogLevel): void;
    /**
     * Encode a QUERY_DISPLAY_GROUPS message.
     */
    queryDisplayGroups(reqId: number): void;
    /**
     * Encode a UPDATE_DISPLAY_GROUP message.
     */
    updateDisplayGroup(reqId: number, contractInfo: string): void;
    /**
     * Encode a SUBSCRIBE_TO_GROUP_EVENTS message.
     */
    subscribeToGroupEvents(reqId: number, groupId: number): void;
    /**
     * Encode a UNSUBSCRIBE_FROM_GROUP_EVENTS message.
     */
    unsubscribeToGroupEvents(reqId: number): void;
    /**
     * Encode a REQ_SEC_DEF_OPT_PARAMS message.
     */
    reqSecDefOptParams(reqId: number, underlyingSymbol: string, futFopExchange: string, underlyingSecType: string, underlyingConId: number): void;
    /**
     * Encode a REQ_SOFT_DOLLAR_TIERS message.
     */
    reqSoftDollarTiers(reqId: number): void;
    /**
     * Encode a REQ_FAMILY_CODES message.
     */
    reqFamilyCodes(): void;
    /**
     * Encode a REQ_MATCHING_SYMBOLS message.
     */
    reqMatchingSymbols(reqId: number, pattern: string): void;
    /**
     * Encode a REQ_MKT_DEPTH_EXCHANGES message.
     */
    reqMktDepthExchanges(): void;
    /**
     * Encode a REQ_SMART_COMPONENTS message.
     */
    reqSmartComponents(reqId: number, bboExchange: string): void;
    /**
     * Encode a REQ_NEWS_ARTICLE message.
     */
    reqNewsArticle(reqId: number, providerCode: string, articleId: string, newsArticleOptions: TagValue[]): void;
    /**
     * Encode a REQ_NEWS_PROVIDERS message.
     */
    reqNewsProviders(): void;
    /**
     * Encode a REQ_HISTORICAL_NEWS message.
     */
    reqHistoricalNews(reqId: number, conId: number, providerCodes: string, startDateTime: string, endDateTime: string, totalResults: number, historicalNewsOptions: TagValue[]): void;
    /**
     * Encode a REQ_HISTOGRAM_DATA message.
     */
    reqHistogramData(reqId: number, contract: Contract, useRTH: boolean, timePeriod: string): void;
    /**
     * Encode a CANCEL_HISTOGRAM_DATA message.
     */
    cancelHistogramData(reqId: number): void;
    /**
     * Encode a CANCEL_HISTOGRAM_DATA message.
     */
    cancelHeadTimestamp(reqId: number): void;
    /**
     * Encode a REQ_MARKET_RULE message.
     */
    reqMarketRule(marketRuleId: number): void;
    /**
     * Encode a REQ_MARKET_RULE message.
     */
    reqCompletedOrders(apiOnly: boolean): void;
    reqWshMetaData(reqId: number): void;
    reqCancelWshMetaData(reqId: number): void;
    reqWshEventData(reqId: number, wshEventData: WshEventData): void;
    reqCancelWshEventData(reqId: number): void;
    reqUserInfo(reqId: number): void;
}
