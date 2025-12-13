import { EventName } from "../../api/data/enum/event-name";
import { IN_MSG_ID } from "./enum/in-msg-id";
/**
 * @internal
 *
 * An underrun error on the input de-serialization.
 */
export declare class UnderrunError extends Error {
    message: string;
    constructor(message?: string);
    readonly stack: string;
    readonly name = "UnderrunError";
}
/**
 * @internal
 *
 * Callback interface of the [[Decoder]].
 */
export interface DecoderCallbacks {
    /** Get the IB API server version. */
    readonly serverVersion: number;
    /**
     * Emit an event to public API interface.
     *
     * @param eventName Event name.
     * @param args Event arguments.
     */
    emitEvent(eventName: EventName, ...args: unknown[]): void;
    /**
     * Emit an error event to public API interface.
     *
     * @param errMsg The error test message.
     * @param code The code identifying the error.
     * @param reqId The request identifier which generated the error.
     * @param advancedOrderReject An object providing more information in case of an order rejection
     */
    emitError(errMsg: string, code: number, reqId?: number, advancedOrderReject?: unknown): void;
    /**
     * Emit an information message event to public API interface.
     *
     * @param message The message text.
     * @param code The message code.
     */
    emitInfo(message: string, code: number): void;
}
/**
 * @internal
 *
 * Class for decoding token data to messages and emitting events it to the
 * [[Controller]] event queue.
 */
export declare class Decoder {
    private callback;
    /**
     * Create an [[Incoming]] object.
     *
     * @param callback A [[DecoderCallbacks]] implementation.
     */
    constructor(callback: DecoderCallbacks);
    /**
     * Input data queue.
     *
     * If the value is a string, this is a tokens as received from TWS / IB Gateway.
     * If the value is undefined, this signals the boundary (start or end) of a message (used with V100 protocol only).
     */
    private dataQueue;
    /** Data emit queue (data to be emitted to controller). */
    private emitQueue;
    /**
     * Add a new message to queue.
     *
     * Used on V100 protocol.
     */
    enqueueMessage(tokens: string[]): void;
    /**
     * Add new tokens to queue.
     *
     * Used on pre-V100 protocol.
     */
    enqueueTokens(tokens: string[]): void;
    /**
     * Process a message on data queue.
     */
    processMsg(msgId: IN_MSG_ID): void;
    /**
     * Process the data queue and emit events.
     */
    process(): void;
    /**
     * Get the API server version.
     */
    private get serverVersion();
    private decodeUnicodeEscapedString;
    /**
     * Read a string token from queue.
     */
    readStr(): string;
    /**
     * Read a token from queue and return it as boolean value.
     */
    readBool(): boolean;
    /**
     * Read a token from queue and return it as boolean value.
     *
     * @deprecated readBool is probably what you are looking for
     */
    readBoolFromInt: () => boolean;
    /**
     * Read a token from queue and return it as floating point value.
     *
     * Returns 0 if the token is empty.
     * Returns undefined is the token is Number.MAX_VALUE.
     */
    readDouble(): number | undefined;
    /**
     * Read a token from queue and return it as floating point value.
     *
     * Returns undefined if the token is empty or is Number.MAX_VALUE.
     */
    readDecimal(): number | undefined;
    /**
     * Read a token from queue and return it as floating point value.
     *
     * Returns undefined if the token is empty or Number.MAX_VALUE.
     */
    readDoubleOrUndefined(): number | undefined;
    /**
     * Read a token from queue and return it as integer value.
     *
     * Returns 0 if the token is empty.
     */
    readInt(): number;
    /**
     * Read a token from queue and return it as integer value.
     *
     * Returns Number.MAX_VALUE if the token is empty.
     * @deprecated readIntOrUndefined is probably what you are looking for
     */
    readIntMax: () => number | undefined;
    /**
     * Read a token from queue and return it as integer value.
     *
     * Returns undefined if the token is empty or `2147483647`.
     */
    readIntOrUndefined(): number | undefined;
    /**
     * Drain all tokens on queue until the start marker of a new message or until queue is empty.
     */
    private drainQueue;
    /**
     * Add tokens to the emit queue.
     */
    private emit;
    /**
     * Decode a TICK_PRICE message from data queue and emit a tickPrice and tickSize event.
     */
    private decodeMsg_TICK_PRICE;
    /**
     * Decode a TICK_SIZE message from data queue and emit an tickSize event.
     */
    private decodeMsg_TICK_SIZE;
    /**
     * Decode a ORDER_STATUS message from data queue and emit an orderStatus event.
     */
    private decodeMsg_ORDER_STATUS;
    /**
     * Decode a ERR_MSG message from data queue and emit and error event.
     */
    private decodeMsg_ERR_MSG;
    /**
     * Decode a OPEN_ORDER message from data queue and emit a openOrder event.
     */
    private decodeMsg_OPEN_ORDER;
    /**
     * Decode a OPEN_ORDER message from data queue and emit a updateAccountValue event.
     */
    private decodeMsg_ACCT_VALUE;
    /**
     * Decode a PORTFOLIO_VALUE message from data queue and emit a updatePortfolio (PortfolioValue) event.
     */
    private decodeMsg_PORTFOLIO_VALUE;
    /**
     * Decode a ACCT_UPDATE_TIME message from data queue and emit a updateAccountTime event.
     */
    private decodeMsg_ACCT_UPDATE_TIME;
    /**
     * Decode a NEXT_VALID_ID message from data queue and emit a nextValidId event.
     */
    private decodeMsg_NEXT_VALID_ID;
    /**
     * Decode a CONTRACT_DATA message from data queue and emit a contractDetails event.
     */
    private decodeMsg_CONTRACT_DATA;
    /**
     * Decode a EXECUTION_DATA message from data queue and emit a execDetails event.
     */
    private decodeMsg_EXECUTION_DATA;
    /**
     * Decode a MARKET_DEPTH message from data queue and emit a MarketDepth event.
     */
    private decodeMsg_MARKET_DEPTH;
    /**
     * Decode a MARKET_DEPTH_L2 message from data queue and emit a MarketDepthL2 event.
     */
    private decodeMsg_MARKET_DEPTH_L2;
    /**
     * Decode a NEWS_BULLETINS message from data queue and emit a updateNewsBulletin event.
     */
    private decodeMsg_NEWS_BULLETINS;
    /**
     * Decode a MANAGED_ACCTS message from data queue and emit a managedAccounts event.
     */
    private decodeMsg_MANAGED_ACCTS;
    /**
     * Decode a RECEIVE_FA message from data queue and emit a receiveFA event.
     */
    private decodeMsg_RECEIVE_FA;
    /**
     * Decode a HISTORICAL_DATA message from data queue and emit historicalData events.
     */
    private decodeMsg_HISTORICAL_DATA;
    /**
     * Decode a HISTORICAL_DATA_UPDATE message from data queue and emit historicalDataUpdate events.
     */
    private decodeMsg_HISTORICAL_DATA_UPDATE;
    /**
     * Decode a REROUTE_MKT_DATA message from data queue and emit a rerouteMktDataReq event.
     */
    private decodeMsg_REROUTE_MKT_DATA;
    /**
     * Decode a REROUTE_MKT_DEPTH message from data queue and emit a rerouteMktDepthReq event.
     */
    private decodeMsg_REROUTE_MKT_DEPTH;
    /**
     * Decode a MARKET_RULE message from data queue and emit a marketRule event.
     */
    private decodeMsg_MARKET_RULE;
    /**
     * Decode a BOND_CONTRACT_DATA message from data queue and emit a BondContractData event.
     */
    private decodeMsg_BOND_CONTRACT_DATA;
    /**
     * Decode a SCANNER_PARAMETERS message from data queue and emit a scannerParameters event.
     */
    private decodeMsg_SCANNER_PARAMETERS;
    /**
     * Decode a SCANNER_DATA message from data queue and emit a scannerData and scannerDataEnd event.
     */
    private decodeMsg_SCANNER_DATA;
    /**
     * Decode a TICK_OPTION_COMPUTATION message from data queue and emit a tickOptionComputation event.
     */
    private decodeMsg_TICK_OPTION_COMPUTATION;
    /**
     * Decode a TICK_GENERIC message from data queue and emit a tickGeneric event.
     */
    private decodeMsg_TICK_GENERIC;
    /**
     * Decode a TICK_STRING message from data queue and emit a tickString event.
     */
    private decodeMsg_TICK_STRING;
    /**
     * Decode a TICK_EFP message from data queue and emit a tickEFP event.
     */
    private decodeMsg_TICK_EFP;
    /**
     * Decode a CURRENT_TIME message from data queue and emit a currentTime event.
     */
    private decodeMsg_CURRENT_TIME;
    /**
     * Decode a REAL_TIME_BARS message from data queue and emit a realtimeBars event.
     */
    private decodeMsg_REAL_TIME_BARS;
    /**
     * Decode a REAL_TIME_BARS message from data queue and emit a fundamentalData event.
     */
    private decodeMsg_FUNDAMENTAL_DATA;
    /**
     * Decode a CONTRACT_DATA_END message from data queue and emit a contractDetailsEnd event.
     */
    private decodeMsg_CONTRACT_DATA_END;
    /**
     * Decode a OPEN_ORDER_END message from data queue and emit a openOrderEnd event.
     */
    private decodeMsg_OPEN_ORDER_END;
    /**
     * Decode a ACCT_DOWNLOAD_END  message from data queue and emit a accountDownloadEnd event.
     */
    private decodeMsg_ACCT_DOWNLOAD_END;
    /**
     * Decode a EXECUTION_DATA_END  message from data queue and emit a execDetailsEnd event.
     */
    private decodeMsg_EXECUTION_DATA_END;
    /**
     * Decode a DELTA_NEUTRAL_VALIDATION message from data queue and emit a deltaNeutralValidation event.
     */
    private decodeMsg_DELTA_NEUTRAL_VALIDATION;
    /**
     * Decode a TICK_SNAPSHOT_END message from data queue and emit a tickSnapshotEnd event.
     */
    private decodeMsg_TICK_SNAPSHOT_END;
    /**
     * Decode a MARKET_DATA_TYPE message from data queue and emit a marketDataType event.
     */
    private decodeMsg_MARKET_DATA_TYPE;
    /**
     * Decode a COMMISSION_REPORT message from data queue and emit a commissionReport event.
     */
    private decodeMsg_COMMISSION_REPORT;
    /**
     * Decode a POSITION message from data queue and emit a position event.
     */
    private decodeMsg_POSITION;
    /**
     * Decode a POSITION_END message from data queue and emit a positionEnd event.
     */
    private decodeMsg_POSITION_END;
    /**
     * Decode a ACCOUNT_SUMMARY message from data queue and emit a accountSummary event.
     */
    private decodeMsg_ACCOUNT_SUMMARY;
    /**
     * Decode a ACCOUNT_SUMMARY message from data queue and emit a accountSummaryEnd event.
     */
    private decodeMsg_ACCOUNT_SUMMARY_END;
    /**
     * Decode a DISPLAY_GROUP_LIST message from data queue and emit a displayGroupList event.
     */
    private decodeMsg_DISPLAY_GROUP_LIST;
    /**
     * Decode a DISPLAY_GROUP_UPDATED message from data queue and emit a displayGroupUpdated event.
     */
    private decodeMsg_DISPLAY_GROUP_UPDATED;
    /**
     * Decode a POSITION_MULTI message from data queue and emit a PositionMulti event.
     */
    private decodeMsg_POSITION_MULTI;
    /**
     * Decode a POSITION_MULTI_END message from data queue and emit a positionMultiEnd event.
     */
    private decodeMsg_POSITION_MULTI_END;
    /**
     * Decode a ACCOUNT_UPDATE_MULTI message from data queue and emit a accountUpdateMulti event.
     */
    private decodeMsg_ACCOUNT_UPDATE_MULTI;
    /**
     * Decode a ACCOUNT_UPDATE_MULTI_END message from data queue and emit a accountUpdateMultiEnd event.
     */
    private decodeMsg_ACCOUNT_UPDATE_MULTI_END;
    /**
     * Decode a SECURITY_DEFINITION_OPTION_PARAMETER message from data queue and emit a securityDefinitionOptionParameter event.
     */
    private decodeMsg_SECURITY_DEFINITION_OPTION_PARAMETER;
    /**
     * Decode a SECURITY_DEFINITION_OPTION_PARAMETER_END message from data queue and emit a securityDefinitionOptionParameterEnd event.
     */
    private decodeMsg_SECURITY_DEFINITION_OPTION_PARAMETER_END;
    /**
     * Decode a SOFT_DOLLAR_TIERS message from data queue and emit a softDollarTiers event.
     */
    private decodeMsg_SOFT_DOLLAR_TIERS;
    /**
     * Decode a FAMILY_CODES message from data queue and emit a familyCodes event.
     */
    private decodeMsg_FAMILY_CODES;
    /**
     * Decode a SYMBOL_SAMPLES message from data queue and emit a symbolSamples event.
     */
    private decodeMsg_SYMBOL_SAMPLES;
    /**
     * Decode a MKT_DEPTH_EXCHANGES message from data queue and emit a mktDepthExchanges event.
     */
    private decodeMsg_MKT_DEPTH_EXCHANGES;
    /**
     * Decode a TICK_REQ_PARAMS message from data queue and emit a tickReqParams event.
     */
    private decodeMsg_TICK_REQ_PARAMS;
    /**
     * Decode a SMART_COMPONENTS message from data queue and emit a smartComponents event.
     */
    private decodeMsg_SMART_COMPONENTS;
    /**
     * Decode a NEWS_ARTICLE message from data queue and emit a newsArticle event.
     */
    private decodeMsg_NEWS_ARTICLE;
    /**
     * Decode a TICK_NEWS message from data queue and emit a tickNews event.
     */
    private decodeMsg_TICK_NEWS;
    /**
     * Decode a NEWS_PROVIDERS message from data queue and emit a newsProviders event.
     */
    private decodeMsg_NEWS_PROVIDERS;
    /**
     * Decode a HISTORICAL_NEWS message from data queue and emit a historicalNews event.
     */
    private decodeMsg_HISTORICAL_NEWS;
    /**
     * Decode a HISTORICAL_NEWS_END message from data queue and emit a historicalNewsEnd event.
     */
    private decodeMsg_HISTORICAL_NEWS_END;
    /**
     * Decode a HEAD_TIMESTAMP message from data queue and emit a headTimestamp event.
     */
    private decodeMsg_HEAD_TIMESTAMP;
    /**
     * Decode a HISTOGRAM_DATA message from data queue and emit a histogramData event.
     */
    private decodeMsg_HISTOGRAM_DATA;
    /**
     * Decode a PNL message from data queue and emit a pnl event.
     */
    private decodeMsg_PNL;
    /**
     * Decode a PNL_SINGLE message from data queue and emit a pnlSingle event.
     */
    private decodeMsg_PNL_SINGLE;
    /**
     * Decode a HISTORICAL_TICKS message from data queue and emit a historicalTicks event.
     */
    private decodeMsg_HISTORICAL_TICKS;
    /**
     * Decode a HISTORICAL_TICKS_BID_ASK message from data queue and emit a historicalTicksBidAsk event.
     */
    private decodeMsg_HISTORICAL_TICKS_BID_ASK;
    /**
     * Decode a HISTORICAL_TICKS_LAST message from data queue and emit a historicalTicksLast event.
     */
    private decodeMsg_HISTORICAL_TICKS_LAST;
    /**
     * Decode a TICK_BY_TICK message from data queue and a emit tickByTickAllLast, tickByTickBidAsk or tickByTickMidPoint event.
     */
    private decodeMsg_TICK_BY_TICK;
    /**
     * Decode a ORDER_BOUND message from data queue and a emit orderBound event.
     */
    private decodeMsg_ORDER_BOUND;
    /**
     * Decode a COMPLETED_ORDER message from data queue and a emit completedOrder event.
     */
    private decodeMsg_COMPLETED_ORDER;
    /**
     * Decode a COMPLETED_ORDER_END message from data queue and a emit completedOrdersEnd event.
     */
    private decodeMsg_COMPLETED_ORDERS_END;
    /**
     * Decode a REPLACE_FA_END message from data queue and a emit replaceFAEnd event.
     */
    private decodeMsg_REPLACE_FA_END;
    /**
     * Decode a WSH_META_DATA message from data queue and a emit wshMetaData event.
     */
    private decodeMsg_WSH_META_DATA;
    /**
     * Decode a WSH_EVENT_DATA message from data queue and a emit wshEventData event.
     */
    private decodeMsg_WSH_EVENT_DATA;
    /**
     * Decode a HISTORICAL_SCHEDULE message from data queue and a emit historicalSchedule event.
     */
    private decodeMsg_HISTORICAL_SCHEDULE;
    /**
     * Decode a USER_INFO message from data queue and a emit userInfo event.
     */
    private decodeMsg_USER_INFO;
    /**
     * Read last trade date, parse it and assign to proper [[ContractDetails]] attributes.
     */
    private readLastTradeDate;
}
