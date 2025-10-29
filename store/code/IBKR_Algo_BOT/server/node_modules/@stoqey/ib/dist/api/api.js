"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApi = exports.MIN_SERVER_VER_SUPPORTED = exports.MAX_SUPPORTED_SERVER_VERSION = void 0;
/*
 * Typescript implementation of the Interactive Brokers TWS (or IB Gateway) API.
 */
/* eslint @typescript-eslint/no-unsafe-declaration-merging:warn,@typescript-eslint/unified-signatures:off */
const eventemitter3_1 = require("eventemitter3");
const controller_1 = require("../core/io/controller");
const wsh_1 = __importDefault(require("./contract/wsh"));
const min_server_version_1 = __importDefault(require("./data/enum/min-server-version"));
/** Maximum supported version. */
exports.MAX_SUPPORTED_SERVER_VERSION = min_server_version_1.default.CME_TAGGING_FIELDS_IN_OPEN_ORDER;
/** Minimum supported version. */
exports.MIN_SERVER_VER_SUPPORTED = 38;
/**
 * Typescript implementation of the Interactive Brokers TWS (or IB Gateway) API.
 *
 * Refer to the official {@link https://interactivebrokers.github.io/tws-api/|Trader Workstation API documentation} for
 * details.
 */
class IBApi extends eventemitter3_1.EventEmitter {
    /**
     * Create a IB API object.
     *
     * @param options Creation options.
     */
    constructor(options) {
        super();
        this.controller = new controller_1.Controller(this, options);
    }
    /**
     * Get the IB API Server version.
     *
     * @see [[MIN_SERVER_VER]]
     */
    get serverVersion() {
        return this.controller.serverVersion;
    }
    /**
     * Returns `true` if currently connected to server, `false` otherwise.
     */
    get isConnected() {
        return this.controller.connected;
    }
    /**
     * Allows to switch between different current (V100+) and previous connection mechanisms.
     *
     * @deprecated pre-V100 support will be removed. Please consider updating your
     * TWS and/or IB Gateway version.
     */
    disableUseV100Plus() {
        return this.controller.disableUseV100Plus();
    }
    /**
     * Connect to the TWS or IB Gateway.
     *
     * @param clientId A unique client id (per TWS or IB Gateway instance).
     * When not specified, the client id from [[IBApiCreationOptions]] or the default client id (0) will used.
     */
    connect(clientId) {
        this.controller.connect(clientId);
        return this;
    }
    /**
     * Disconnect from the TWS or IB Gateway.
     */
    disconnect() {
        this.controller.disconnect();
        return this;
    }
    /**
     * Calculate the volatility for an option.
     * Request the calculation of the implied volatility based on hypothetical option and its underlying prices.
     * The calculation will be emitted as tickOptionComputation event.
     *
     * @param reqId Unique identifier of the request.
     * @param contract The option's contract for which the volatility wants to be calculated.
     * @param optionPrice Hypothetical option price.
     * @param underPrice Hypothetical option's underlying price.
     *
     * @see [[cancelCalculateImpliedVolatility]]
     */
    calculateImpliedVolatility(reqId, contract, optionPrice, underPrice) {
        this.controller.schedule(() => this.controller.encoder.calculateImpliedVolatility(reqId, contract, optionPrice, underPrice));
        return this;
    }
    /**
     * Calculates an option's price based on the provided volatility and its underlying price.
     * The calculation will be emitted as tickOptionComputation event.
     *
     * @param reqId Unique identifier of the request.
     * @param contract The option's contract for which the price wants to be calculated.
     * @param volatility Hypothetical volatility.
     * @param underPrice Hypothetical underlying price.
     *
     * @see [[cancelCalculateOptionPrice]]
     */
    calculateOptionPrice(reqId, contract, volatility, underPrice) {
        this.controller.schedule(() => this.controller.encoder.calculateOptionPrice(reqId, contract, volatility, underPrice));
        return this;
    }
    /**
     * Cancels the account's summary request.
     * After requesting an account's summary, invoke this function to cancel it.
     *
     * @param reqId The identifier of the previously performed account request.
     *
     * @see [[reqAccountSummary]]
     */
    cancelAccountSummary(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelAccountSummary(reqId));
        return this;
    }
    /**
     * Cancels account updates request for account and/or model.
     *
     * @param reqId Account subscription to cancel.
     *
     * @see [[reqAccountUpdatesMulti]]
     */
    cancelAccountUpdatesMulti(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelAccountUpdatesMulti(reqId));
        return this;
    }
    /**
     * Cancels an option's implied volatility calculation request.
     *
     * @param reqId The identifier of the implied volatility's calculation request.
     *
     * @see [[calculateImpliedVolatility]]
     */
    cancelCalculateImpliedVolatility(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelCalculateImpliedVolatility(reqId));
        return this;
    }
    /**
     * Cancels an option's price calculation request.
     *
     * @param reqId The identifier of the option's price's calculation request.
     *
     * @see [[calculateOptionPrice]]
     */
    cancelCalculateOptionPrice(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelCalculateOptionPrice(reqId));
        return this;
    }
    /**
     * Cancels Fundamental data request.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqFundamentalData]]
     */
    cancelFundamentalData(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelFundamentalData(reqId));
        return this;
    }
    /**
     * Cancels a pending [[reqHeadTimeStamp]] request.
     *
     * @param reqId Id of the request
     *
     * @see [[reqHeadTimeStamp]]
     */
    cancelHeadTimestamp(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelHeadTimestamp(reqId));
        return this;
    }
    /**
     * Cancels a historical data request.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqHistogramData]]
     */
    cancelHistogramData(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelHistogramData(reqId));
        return this;
    }
    /**
     * Cancels a historical data request.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqHistoricalData]]
     */
    cancelHistoricalData(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelHistoricalData(reqId));
        return this;
    }
    /**
     * Cancels a RT Market Data request.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqMktData]]
     */
    cancelMktData(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelMktData(reqId));
        return this;
    }
    /**
     * Cancel a market depth's request.
     *
     * @param reqId The request's identifier.
     * @param isSmartDepth TODO document
     *
     * @see [[reqMktDepth]]
     */
    cancelMktDepth(reqId, isSmartDepth) {
        this.controller.schedule(() => this.controller.encoder.cancelMktDepth(reqId, isSmartDepth));
        return this;
    }
    /**
     * Cancels IB's news bulletin subscription.
     *
     * @see [[reqNewsBulletins]]
     */
    cancelNewsBulletins() {
        this.controller.schedule(() => this.controller.encoder.cancelNewsBulletins());
        return this;
    }
    /**
     * Cancels an active order placed by from the same API client ID.
     *
     * Note: API clients cannot cancel individual orders placed by other clients.
     * Use [[reqGlobalCancel]] instead.
     *
     * @param orderId Specify which order should be cancelled by its identifier.
     * @param orderCancel Specify the time the order should be cancelled. An empty string will cancel the order immediately.
     *
     * @see [[placeOrder]], [[reqGlobalCancel]]
     */
    cancelOrder(orderId, orderCancelParam) {
        let orderCancel;
        if (orderCancelParam == undefined)
            orderCancel = {
                manualOrderCancelTime: undefined,
                extOperator: "",
                manualOrderIndicator: undefined,
            };
        else if (typeof orderCancelParam == "string")
            orderCancel = {
                manualOrderCancelTime: orderCancelParam,
                extOperator: "",
                manualOrderIndicator: undefined,
            };
        else
            orderCancel = orderCancelParam;
        this.controller.schedule(() => this.controller.encoder.cancelOrder(orderId, orderCancel));
        return this;
    }
    /**
     * Cancels subscription for real time updated daily PnL.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqPnL]]
     */
    cancelPnL(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelPnL(reqId));
        return this;
    }
    /**
     * Cancels real time subscription for a positions daily PnL information.
     *
     * @param reqId The request's identifier.
     *
     * @see [[reqPnLSingle]]
     */
    cancelPnLSingle(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelPnLSingle(reqId));
        return this;
    }
    /**
     * Cancels a previous position subscription request made with reqPositions.
     *
     * @see [[reqPositions]]
     */
    cancelPositions() {
        this.controller.schedule(() => this.controller.encoder.cancelPositions());
        return this;
    }
    /**
     * Cancels positions request for account and/or model.
     *
     * @param reqId The identifier of the request to be canceled.
     *
     * @see [[reqPositionsMulti]]
     */
    cancelPositionsMulti(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelPositionsMulti(reqId));
        return this;
    }
    /**
     * Cancels Real Time Bars' subscription.
     *
     * @param reqId The request's identifier.
     */
    cancelRealTimeBars(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelRealTimeBars(reqId));
        return this;
    }
    /**
     * Cancels Scanner Subscription.
     *
     * @param reqId The subscription's unique identifier.
     *
     * @see [[reqScannerSubscription]], [[reqScannerParameters]]
     */
    cancelScannerSubscription(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelScannerSubscription(reqId));
        return this;
    }
    /**
     * Cancels tick-by-tick data.
     *
     * @param reqId Unique identifier of the request.
     *
     * @see [[reqTickByTickData]]
     */
    cancelTickByTickData(reqId) {
        this.controller.schedule(() => this.controller.encoder.cancelTickByTickData(reqId));
        return this;
    }
    /**
     * Exercises an options contract.
     *
     * Note: this function is affected by a TWS setting which specifies if an exercise request must be finalized.
     *
     * @param reqId The exercise request's identifier.
     * @param contract The option [[Contract]] to be exercised.
     * @param exerciseAction 1 to exercise the option, 2 to let the option lapse.
     * @param exerciseQuantity Number of contracts to be exercised.
     * @param account Destination account.
     * @param override Specifies whether your setting will override the system's natural action.
     * For example, if your action is "exercise" and the option is not in-the-money,
     * by natural action the option would not exercise.
     * If you have override set to "yes" the natural action would be overridden and the out-of-the money option would be
     *   exercised. Set to 1 to override, set to 0 not to.
     */
    exerciseOptions(reqId, contract, exerciseAction, exerciseQuantity, account, override) {
        this.controller.schedule(() => this.controller.encoder.exerciseOptions(reqId, contract, exerciseAction, exerciseQuantity, account, override));
        return this;
    }
    /**
     * Places or modifies an order.
     * @param id The order's unique identifier.
     * Use a sequential id starting with the id received at the nextValidId method.
     * If a new order is placed with an order ID less than or equal to the order ID of a previous order an error will
     *   occur.
     * @param contract The order's [[Contract]].
     * @param order The [[Order]] object.
     *
     * @see [[reqAllOpenOrders]], [[reqAutoOpenOrders]], [[reqOpenOrders]], [[cancelOrder]], [[reqGlobalCancel]]
     */
    placeOrder(id, contract, order) {
        this.controller.schedule(() => this.controller.encoder.placeOrder(id, contract, order));
        return this;
    }
    /**
     * Requests all available Display Groups in TWS.
     *
     * @param reqId The ID of this request.
     */
    queryDisplayGroups(reqId) {
        this.controller.schedule(() => this.controller.encoder.queryDisplayGroups(reqId));
        return this;
    }
    /**
     * Replaces Financial Advisor's settings A Financial Advisor can define three different configurations:
     *
     * - Groups: offer traders a way to create a group of accounts and apply a single allocation method to all accounts
     * in the group.
     * - Profiles: let you allocate shares on an account-by-account basis using a predefined calculation value.
     * - Account Aliases: let you easily identify the accounts by meaningful names rather than account numbers.
     * More information at
     * https://www.interactivebrokers.com/en/?f=%2Fen%2Fsoftware%2Fpdfhighlights%2FPDF-AdvisorAllocations.php%3Fib_entity%3Dllc
     *
     * @param faDataType The configuration to change.
     * @param xml Zhe xml-formatted configuration string.
     */
    replaceFA(reqId, faDataType, xml) {
        this.controller.schedule(() => this.controller.encoder.replaceFA(reqId, faDataType, xml));
        return this;
    }
    /**
     * Requests a specific account's summary.
     * This method will subscribe to the account summary as presented in the TWS' Account Summary tab.
     * The data is emitted as accountSummary event.
     *
     * https://www.interactivebrokers.com/en/software/tws/accountwindowtop.htm
     *
     * @param reqId The unique request identifier.
     * @param group Set to "All" to return account summary data for all accounts,
     * or set to a specific Advisor Account Group name that has already been created in TWS Global Configuration.
     * @param tags A comma separated list with the desired tags:
     * - AccountType — Identifies the IB account structure
     * - NetLiquidation — The basis for determining the price of the assets in your account. Total cash value + stock
     *   value + options value + bond value
     * - TotalCashValue — Total cash balance recognized at the time of trade + futures PNL
     * - SettledCash — Cash recognized at the time of settlement - purchases at the time of trade - commissions - taxes -
     *   fees
     * - AccruedCash — Total accrued cash value of stock, commodities and securities
     * - BuyingPower — Buying power serves as a measurement of the dollar value of securities that one may purchase in a
     *   securities account without depositing additional funds
     * - EquityWithLoanValue — Forms the basis for determining whether a client has the necessary assets to either
     *   initiate or maintain security positions. Cash + stocks + bonds + mutual funds
     * - PreviousDayEquityWithLoanValue — Marginable Equity with Loan value as of 16:00 ET the previous day
     * - GrossPositionValue — The sum of the absolute value of all stock and equity option positions
     * - RegTEquity — Regulation T equity for universal account
     * - RegTMargin — Regulation T margin for universal account
     * - SMA — Special Memorandum Account: Line of credit created when the market value of securities in a Regulation T
     *   account increase in value
     * - InitMarginReq — Initial Margin requirement of whole portfolio
     * - MaintMarginReq — Maintenance Margin requirement of whole portfolio
     * - AvailableFunds — This value tells what you have available for trading
     * - ExcessLiquidity — This value shows your margin cushion, before liquidation
     * - Cushion — Excess liquidity as a percentage of net liquidation value
     * - FullInitMarginReq — Initial Margin of whole portfolio with no discounts or intraday credits
     * - FullMaintMarginReq — Maintenance Margin of whole portfolio with no discounts or intraday credits
     * - FullAvailableFunds — Available funds of whole portfolio with no discounts or intraday credits
     * - FullExcessLiquidity — Excess liquidity of whole portfolio with no discounts or intraday credits
     * - LookAheadNextChange — Time when look-ahead values take effect
     * - LookAheadInitMarginReq — Initial Margin requirement of whole portfolio as of next period's margin change
     * - LookAheadMaintMarginReq — Maintenance Margin requirement of whole portfolio as of next period's margin change
     * - LookAheadAvailableFunds — This value reflects your available funds at the next margin change
     * - LookAheadExcessLiquidity — This value reflects your excess liquidity at the next margin change
     * - HighestSeverity — A measure of how close the account is to liquidation
     * - DayTradesRemaining — The Number of Open/Close trades a user could put on before Pattern Day Trading is detected.
     *   A value of "-1" means that the user can put on unlimited day trades.
     * - Leverage — GrossPositionValue / NetLiquidation
     * - $LEDGER — Single flag to relay all cash balance tags*, only in base currency.
     * - $LEDGER:CURRENCY — Single flag to relay all cash balance tags*, only in the specified currency.
     * - $LEDGER:ALL — Single flag to relay all cash balance tags* in all currencies.
     *
     * @see [[cancelAccountSummary]]
     */
    reqAccountSummary(reqId, group, tags) {
        this.controller.schedule(() => this.controller.encoder.reqAccountSummary(reqId, group, tags));
        return this;
    }
    /**
     * Subscribes to a specific account's information and portfolio.
     * Through this method, a single account's subscription can be started/stopped.
     * As a result from the subscription, the account's information, portfolio and last update time will be emitted as
     * updateAccountValue, updateAccountPortfolio, updateAccountTime event respectively.
     *
     * All account values and positions will be returned initially, and then there will only be updates when there is a
     * change in a position, or to an account value every 3 minutes if it has changed.
     *
     * Only one account can be subscribed at a time.
     *
     * A second subscription request for another account when the previous one is still active will cause the first one
     * to be canceled in favour of the second one. Consider user reqPositions if you want to retrieve all your accounts'
     * portfolios directly.
     *
     * @param subscribe Set to true to start the subscription and to false to stop it.
     * @param acctCode The account id (i.e. U123456) for which the information is requested.
     *
     * @see [[reqPositions]]
     */
    reqAccountUpdates(subscribe, acctCode) {
        this.controller.schedule(() => this.controller.encoder.reqAccountUpdates(subscribe, acctCode));
        return this;
    }
    /**
     * Requests account updates for account and/or model.
     *
     * @param reqId Identifier to label the request.
     * @param acctCode Account values can be requested for a particular account
     * @param modelCode Values can also be requested for a model
     * @param ledgerAndNLV returns light-weight request; only currency positions as opposed to account values and
     *   currency positions.
     *
     * @see [[cancelAccountUpdatesMulti]]
     */
    reqAccountUpdatesMulti(reqId, acctCode, modelCode, ledgerAndNLV) {
        this.controller.schedule(() => this.controller.encoder.reqAccountUpdatesMulti(reqId, acctCode, modelCode, ledgerAndNLV));
        return this;
    }
    /**
     * Requests all current open orders in associated accounts at the current moment.
     * The existing orders will be received via the openOrder and orderStatus events.
     *
     * Open orders are returned once; this function does not initiate a subscription.
     *
     * @see [[reqAutoOpenOrders]], [[reqOpenOrders]]
     */
    reqAllOpenOrders() {
        this.controller.schedule(() => this.controller.encoder.reqAllOpenOrders());
        return this;
    }
    /**
     * Requests status updates about future orders placed from TWS. Can only be used with client ID 0.
     *
     * @param bAutoBind if set to `true`, the newly created orders will be assigned an API order ID and implicitly
     *   associated with this client. If set to `false, future orders will not be.
     *
     * @see [[reqAllOpenOrders]], [[reqOpenOrders]], [[cancelOrder]], [[reqGlobalCancel]]
     */
    reqAutoOpenOrders(bAutoBind) {
        this.controller.schedule(() => this.controller.encoder.reqAutoOpenOrders(bAutoBind));
        return this;
    }
    /**
     * Requests completed orders.
     *
     * @param apiOnly Request only API orders.
     */
    reqCompletedOrders(apiOnly) {
        this.controller.schedule(() => this.controller.encoder.reqCompletedOrders(apiOnly));
        return this;
    }
    /**
     * Requests metadata from the WSH calendar.
     *
     * @param reqId The unique request identifier.
     *
     * @see [[reqCancelWshMetaData]]
     */
    reqWshMetaData(reqId) {
        this.controller.schedule(() => this.controller.encoder.reqWshMetaData(reqId));
        return this;
    }
    /**
     * Cancels pending request for WSH metadata.
     *
     * @param reqId The unique request identifier.
     */
    reqCancelWshMetaData(reqId) {
        this.controller.schedule(() => this.controller.encoder.reqCancelWshMetaData(reqId));
        return this;
    }
    /**
     * Requests event data from the wSH calendar.
     *
     * @param reqId The unique request identifier.
     * @param wshEventData Contract id (conId) of ticker or WshEventData describing wanted events.
     *
     * @see [[reqCancelWshEventData]]
     */
    reqWshEventData(reqId, wshEventData) {
        let wshEventData2;
        if (typeof wshEventData == "number")
            wshEventData2 = new wsh_1.default(wshEventData);
        else
            wshEventData2 = wshEventData;
        this.controller.schedule(() => this.controller.encoder.reqWshEventData(reqId, wshEventData2));
        return this;
    }
    /**
     * Cancels pending WSH event data request.
     *
     * @param reqId The unique request identifier.
     */
    reqCancelWshEventData(reqId) {
        this.controller.schedule(() => this.controller.encoder.reqCancelWshEventData(reqId));
        return this;
    }
    /**
     * Requests contract information.
     * This method will provide all the contracts matching the contract provided.
     *
     * It can also be used to retrieve complete options and futures chains.
     * Though it is now (in API version > 9.72.12) advised to use reqSecDefOptParams for that purpose.
     *
     * This information will be emitted as contractDetails event.
     *
     * @param reqId The unique request identifier.
     * @param contract The contract used as sample to query the available contracts.
     */
    reqContractDetails(reqId, contract) {
        this.controller.schedule(() => this.controller.encoder.reqContractDetails(reqId, contract));
        return this;
    }
    /**
     * Requests TWS's current time.
     */
    reqCurrentTime() {
        this.controller.schedule(() => this.controller.encoder.reqCurrentTime());
        return this;
    }
    /**
     * Requests current day's (since midnight) executions matching the filter.
     * Only the current day's executions can be retrieved.
     * Along with the executions, the CommissionReport will also be returned.
     *
     * The execution details will be emitted as execDetails event.
     *
     * @param reqId The request's unique identifier.
     * @param filter The filter criteria used to determine which execution reports are returned.
     */
    reqExecutions(reqId, filter) {
        this.controller.schedule(() => this.controller.encoder.reqExecutions(reqId, filter));
        return this;
    }
    /**
     * Requests family codes for an account, for instance if it is a FA, IBroker, or associated account.
     */
    reqFamilyCodes() {
        this.controller.schedule(() => this.controller.encoder.reqFamilyCodes());
        return this;
    }
    /**
     * Requests the contract's fundamental data. Fundamental data is emitted as fundamentalData event.
     *
     * @param reqId The request's unique identifier.
     * @param contract The contract's description for which the data will be returned.
     * @param reportType there are three available report types:
     * - ReportSnapshot: Company overview.
     * - ReportsFinSummary: Financial summary.
     * - ReportRatios: Financial ratios.
     * - ReportsFinStatements: Financial statements.
     * - RESC: Analyst estimates.
     */
    reqFundamentalData(reqId, contract, reportType, fundamentalDataOptions = []) {
        this.controller.schedule(() => this.controller.encoder.reqFundamentalData(reqId, contract, reportType, fundamentalDataOptions));
        return this;
    }
    /**
     * Cancels all active orders.
     * This method will cancel ALL open orders including those placed directly from TWS.
     *
     * @see [[cancelOrder]]
     */
    reqGlobalCancel(orderCancel) {
        this.controller.schedule(() => this.controller.encoder.reqGlobalCancel(orderCancel || {
            manualOrderCancelTime: undefined,
            extOperator: "",
            manualOrderIndicator: undefined,
        }));
        return this;
    }
    /**
     * Returns the timestamp of earliest available historical data for a contract and data type.
     *
     * @param reqId An identifier for the request.
     * @param contract [[Contract]] object for which head timestamp is being requested.
     * @param whatToShow Type of data for head timestamp - "BID", "ASK", "TRADES", etc
     * @param useRTH Use regular trading hours only, `true` for yes or `false` for no.
     * @param formatDate Set to 1 to obtain the bars' time as yyyyMMdd HH:mm:ss, set to 2 to obtain it like system time
     *   format in seconds.
     */
    reqHeadTimestamp(reqId, contract, whatToShow, useRTH, formatDate) {
        this.controller.schedule(() => this.controller.encoder.reqHeadTimestamp(reqId, contract, whatToShow, useRTH, formatDate));
        return this;
    }
    /**
     * Returns data histogram of specified contract.
     *
     * @param reqId An identifier for the request.
     * @param contract [[Contract]] object for which histogram is being requested
     * @param useRTH Use regular trading hours only, `true` for yes or `false` for no.
     * @param period Period of which data is being requested
     * @param periodUnit Unit of the period of which data is being requested
     */
    reqHistogramData(reqId, contract, useRTH, period, periodUnit) {
        const periodStr = period + " " + periodUnit.toString().toLowerCase() + "s";
        this.controller.schedule(() => this.controller.encoder.reqHistogramData(reqId, contract, useRTH, periodStr));
        return this;
    }
    /**
     * Requests contracts' historical data. When requesting historical data, a finishing time and date is required along
     * with a duration string. For example, having:
     * ````- endDateTime: 20130701 23:59:59 GMT```
     * ````- durationStr: 3 ```
     * will return three days of data counting backwards from July 1st 2013 at 23:59:59 GMT resulting in all the
     * available bars of the last three days until the date and time specified.
     *
     * It is possible to specify a timezone optionally.
     *
     * The resulting bars will be emitted as historicalData event.
     *
     * @param reqId The request's unique identifier.
     * @param contract The contract for which we want to retrieve the data.
     * @param endDateTime Request's ending time with format yyyyMMdd HH:mm:ss {TMZ}
     * @param durationStr The amount of time for which the data needs to be retrieved (number space unit).
     * Note: min duration is "30 S", available units:
     * - S (seconds)
     * - D (days)
     * - W (weeks)
     * - M (months)
     * - Y (years)
     * @param barSizeSetting the size of the bar:
     * - 1 secs
     * - 5 secs
     * - 15 secs
     * - 30 secs
     * - 1 min
     * - 2 mins
     * - 3 mins
     * - 5 mins
     * - 15 mins
     * - 30 mins
     * - 1 hour
     * - 1 day
     * @param whatToShow the kind of information being retrieved:
     * - TRADES
     * - MIDPOINT
     * - BID
     * - ASK
     * - BID_ASK
     * - HISTORICAL_VOLATILITY
     * - OPTION_IMPLIED_VOLATILITY
     * - FEE_RATE
     * - REBATE_RATE
     * @param useRTH Set to `false` to obtain the data which was also generated outside of the Regular Trading Hours, set to `true`
     *   to obtain only the RTH data
     * @param formatDate Set to 1 to obtain the bars' time as yyyyMMdd HH:mm:ss, set to 2 to obtain it like system time
     *   format in seconds
     * @param keepUpToDate Set to `true` to received continuous updates on most recent bar data. If `true`, and
     *   endDateTime cannot be specified.
     */
    reqHistoricalData(reqId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate) {
        this.controller.schedule(() => this.controller.encoder.reqHistoricalData(reqId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate));
        return this;
    }
    /**
     * Requests historical news headlines.
     *
     * @param reqId ID of the request.
     * @param conId Contract id of ticker.
     * @param providerCodes A '+'-separated list of provider codes .
     * @param startDateTime Marks the (exclusive) start of the date range. The format is yyyy-MM-dd HH:mm:ss.0
     * @param endDateTime Marks the (inclusive) end of the date range. The format is yyyy-MM-dd HH:mm:ss.0
     * @param totalResults The maximum number of headlines to fetch (1 - 300).
     */
    reqHistoricalNews(reqId, conId, providerCodes, startDateTime, endDateTime, totalResults, historicalNewsOptions = []) {
        this.controller.schedule(() => this.controller.encoder.reqHistoricalNews(reqId, conId, providerCodes, startDateTime, endDateTime, totalResults, historicalNewsOptions));
        return this;
    }
    /**
     * Requests historical Time&Sales data for an instrument.
     *
     * @param reqId ID of the request.
     * @param contract [[Contract]] object that is subject of query
     * @param startDateTime "20170701 12:01:00". Uses TWS timezone specified at login.
     * @param endDateTime "20170701 13:01:00". In TWS timezone. Exactly one of start time and end time has to be defined.
     * @param numberOfTicks Number of distinct data points. Max currently 1000 per request.
     * @param whatToShow 	(Bid_Ask, Midpoint, Trades) Type of data requested.
     * @param useRTH Data from regular trading hours (1), or all available hours (0)
     * @param ignoreSize A filter only used when the source price is Bid_Ask
     */
    reqHistoricalTicks(reqId, contract, startDateTime, endDateTime, numberOfTicks, whatToShow, useRTH, ignoreSize) {
        this.controller.schedule(() => this.controller.encoder.reqHistoricalTicks(reqId, contract, startDateTime, endDateTime, numberOfTicks, whatToShow, useRTH, ignoreSize));
        return this;
    }
    /**
     * Requests the next valid order ID at the current moment.
     *
     * @param numIds deprecated- this parameter will not affect the value returned to nextValidId
     */
    reqIds(numIds = 0) {
        this.controller.schedule(() => this.controller.encoder.reqIds(numIds));
        return this;
    }
    /**
     * Requests the accounts to which the logged user has access to.
     */
    reqManagedAccts() {
        this.controller.schedule(() => this.controller.encoder.reqManagedAccts());
        return this;
    }
    /**
     * Switches data type returned from reqMktData request to "frozen", "delayed" or "delayed-frozen" market data.
     * Requires TWS/IBG v963+.
     *
     * The API can receive frozen market data from Trader Workstation.
     * Frozen market data is the last data recorded in our system.
     * During normal trading hours, the API receives real-time market data.
     * Invoking this function with argument 2 requests a switch to frozen data immediately or after the close.
     * When the market reopens, the market data type will automatically switch back to real time if available.
     *
     * @param marketDataType By default only real-time (1) market data is enabled.
     * - 1 (real-time) disables frozen, delayed and delayed-frozen market data.
     * - 2 (frozen) enables frozen market data.
     * - 3 (delayed) enables delayed and disables delayed-frozen market data.
     * - 4 (delayed-frozen) enables delayed and delayed-frozen market data.
     */
    reqMarketDataType(marketDataType) {
        this.controller.schedule(() => this.controller.encoder.reqMarketDataType(marketDataType));
        return this;
    }
    /**
     * Requests the contract's market depth (order book).
     *
     * This request must be direct-routed to an exchange and not smart-routed.
     *
     * The number of simultaneous market depth requests allowed in an account is calculated based on a formula
     * that looks at an accounts equity, commissions, and quote booster packs.
     *
     * @param reqId The request's identifier.
     * @param contract The [[Contract]] for which the depth is being requested.
     * @param numRows The number of rows on each side of the order book.
     * @param isSmartDepth Flag indicates that this is smart depth request.
     * @param mktDepthOptions TODO document
     *
     * @see [[cancelMktDepth]]
     */
    reqMktDepth(reqId, contract, numRows, isSmartDepth, mktDepthOptions) {
        this.controller.schedule(() => this.controller.encoder.reqMktDepth(reqId, contract, numRows, isSmartDepth, mktDepthOptions));
        return this;
    }
    /**
     * Requests details about a given market rule.
     * The market rule for an instrument on a particular exchange provides details about how the minimum price increment
     * changes with price. A list of market rule ids can be obtained by invoking reqContractDetails on a particular
     * contract. The returned market rule ID list will provide the market rule ID for the instrument in the correspond
     * valid exchange list in contractDetails.
     *
     * @param marketRuleId The id of market rule.
     */
    reqMarketRule(marketRuleId) {
        this.controller.schedule(() => this.controller.encoder.reqMarketRule(marketRuleId));
        return this;
    }
    /**
     * Requests matching stock symbols.
     *
     * Thr result will be emitted as symbolSamples event.
     *
     * @param reqId ID to specify the request
     * @param pattern Either start of ticker symbol or (for larger strings) company name.
     */
    reqMatchingSymbols(reqId, pattern) {
        this.controller.schedule(() => this.controller.encoder.reqMatchingSymbols(reqId, pattern));
        return this;
    }
    /**
     * Requests real time market data.
     * Returns market data for an instrument either in real time or 10-15 minutes delayed (depending on the market data
     * type specified).
     *
     * @param reqId The request's identifier.
     * @param contract The [[Contract]] for which the data is being requested
     * @param genericTickList comma  separated ids of the available generic ticks:
     * - 100 Option Volume (currently for stocks)
     * - 101 Option Open Interest (currently for stocks)
     * - 104 Historical Volatility (currently for stocks)
     * - 105 Average Option Volume (currently for stocks)
     * - 106 Option Implied Volatility (currently for stocks)
     * - 162 Index Future Premium
     * - 165 Miscellaneous Stats
     * - 221 Mark Price (used in TWS P&L computations)
     * - 225 Auction values (volume, price and imbalance)
     * - 233 RTVolume - contains the last trade price, last trade size, last trade time, total volume, VWAP, and single
     *   trade flag.
     * - 236 Shortable
     * - 256 Inventory
     * - 258 Fundamental Ratios
     * - 411 Realtime Historical Volatility
     * - 456 IBDividends
     * @param snapshot For users with corresponding real time market data subscriptions.
     * A `true` value will return a one-time snapshot, while a `false` value will provide streaming data.
     * @param regulatorySnapshot Snapshot for US stocks requests NBBO snapshots for users which have "US Securities
     *   Snapshot Bundle" subscription but not corresponding Network A, B, or C subscription necessary for streaming *
     *   market data. One-time snapshot of current market price that will incur a fee of 1 cent to the account per
     *   snapshot.
     *
     * @see [[cancelMktData]]
     */
    reqMktData(reqId, contract, genericTickList, snapshot, regulatorySnapshot) {
        this.controller.schedule(() => this.controller.encoder.reqMktData(reqId, contract, genericTickList, snapshot, regulatorySnapshot));
        return this;
    }
    /**
     * Requests venues for which market data is returned to updateMktDepthL2 (those with market makers)
     */
    reqMktDepthExchanges() {
        this.controller.schedule(() => this.controller.encoder.reqMktDepthExchanges());
        return this;
    }
    /**
     * Requests news article body given articleId.
     *
     * @param requestId ID of the request.
     * @param providerCode Short code indicating news provider, e.g. FLY
     * @param articleId ID of the specific article.
     */
    reqNewsArticle(requestId, providerCode, articleId, newsArticleOptions = []) {
        this.controller.schedule(() => this.controller.encoder.reqNewsArticle(requestId, providerCode, articleId, newsArticleOptions));
        return this;
    }
    /**
     * Subscribes to IB's News Bulletins.
     *
     * @param allMsgs If set to `true, will return all the existing bulletins for the current day.
     * If set to `false` to receive only the new bulletins.
     *
     * @see [[cancelNewsBulletin]]
     */
    reqNewsBulletins(allMsgs) {
        this.controller.schedule(() => this.controller.encoder.reqNewsBulletins(allMsgs));
        return this;
    }
    /**
     * Requests news providers which the user has subscribed to.
     */
    reqNewsProviders() {
        this.controller.schedule(() => this.controller.encoder.reqNewsProviders());
        return this;
    }
    /**
     * Requests all open orders placed by this specific API client (identified by the API client id).
     * For client ID 0, this will bind previous manual TWS orders.
     */
    reqOpenOrders() {
        this.controller.schedule(() => this.controller.encoder.reqOpenOrders());
        return this;
    }
    /**
     * Creates subscription for real time daily PnL and unrealized PnL updates.
     *
     * @param reqId ID of the request.
     * @param account Account for which to receive PnL updates.
     * @param modelCode Specify to request PnL updates for a specific model.
     */
    reqPnL(reqId, account, modelCode) {
        this.controller.schedule(() => this.controller.encoder.reqPnL(reqId, account, modelCode ?? null));
        return this;
    }
    /**
     * Requests real time updates for daily PnL of individual positions.
     *
     * @param reqId ID of the request.
     * @param account Account in which position exists.
     * @param modelCode Model in which position exists.
     * @param conId Contract ID (conId) of contract to receive daily PnL updates for.
     * Note: does not return message if invalid conId is entered.
     */
    reqPnLSingle(reqId, account, modelCode, conId) {
        this.controller.schedule(() => this.controller.encoder.reqPnLSingle(reqId, account, modelCode, conId));
        return this;
    }
    /**
     * Subscribes to position updates for all accessible accounts.
     * All positions sent initially, and then only updates as positions change.
     *
     * @see [[cancelPositions]]
     */
    reqPositions() {
        this.controller.schedule(() => this.controller.encoder.reqPositions());
        return this;
    }
    /**
     * Requests position subscription for account and/or model.
     * Initially all positions are returned and then updates are returned for any position changes in real time.
     *
     * @param reqId Request's identifier.
     * @param account If an account Id is provided, only the account's positions belonging to the specified model will be
     *   delivered.
     * @param modelCode The code of the model's positions we are interested in.
     *
     * @see [[cancelPositionsMulti]]
     */
    reqPositionsMulti(reqId, account, modelCode) {
        this.controller.schedule(() => this.controller.encoder.reqPositionsMulti(reqId, account, modelCode));
        return this;
    }
    /**
     * Requests real time bars.
     *
     * Currently, only 5 seconds bars are provided.
     *
     * This request is subject to the same pacing as any historical data request: no more than 60 API queries in more
     * than 600 seconds. Real time bars subscriptions are also included in the calculation of the number of Level 1
     * market data subscriptions allowed in an account.
     *
     * @param reqId The request's unique identifier.
     * @param contract The [[Contract]] for which the depth is being requested
     * @param barSize currently being ignored
     * @param whatToShow the nature of the data being retrieved:
     * - TRADES
     * - MIDPOINT
     * - BID
     * - ASK
     * @param useRTH Set to `false` to obtain the data which was also generated ourside of the Regular Trading Hours.
     * Set to `true` to obtain only the RTH data
     *
     * @see [[cancelRealTimeBars]]
     */
    reqRealTimeBars(reqId, contract, barSize, whatToShow, useRTH, realTimeBarsOptions = []) {
        this.controller.schedule(() => this.controller.encoder.reqRealTimeBars(reqId, contract, barSize, whatToShow, useRTH, realTimeBarsOptions));
        return this;
    }
    /**
     * Requests an XML list of scanner parameters valid in TWS.
     *
     * Not all parameters are valid from API scanner.
     *
     * @sse [[reqScannerSubscription]]
     */
    reqScannerParameters() {
        this.controller.schedule(() => this.controller.encoder.reqScannerParameters());
        return this;
    }
    /**
     * Starts a subscription to market scan results based on the provided parameters.
     *
     * @param reqId The request's identifier.
     * @param subscription Summary of the scanner subscription including its filters.
     * @param scannerSubscriptionOptions TODO document
     * @param scannerSubscriptionFilterOptions TODO document
     *
     * @see [[reqScannerParameters]]
     */
    reqScannerSubscription(reqId, subscription, scannerSubscriptionOptions = [], scannerSubscriptionFilterOptions = []) {
        this.controller.schedule(() => this.controller.encoder.reqScannerSubscription(reqId, subscription, scannerSubscriptionOptions, scannerSubscriptionFilterOptions));
        return this;
    }
    /**
     * Requests security definition option parameters for viewing a contract's option chain.
     *
     * @param reqId The request's identifier.
     * @param underlyingSymbol Underlying symbol name.
     * @param futFopExchange The exchange on which the returned options are trading.
     * Can be set to the empty string "" for all exchanges.
     * @param underlyingSecType The type of the underlying security, i.e. STK
     * @param underlyingConId the contract ID of the underlying security
     */
    reqSecDefOptParams(reqId, underlyingSymbol, futFopExchange, underlyingSecType, underlyingConId) {
        this.controller.schedule(() => this.controller.encoder.reqSecDefOptParams(reqId, underlyingSymbol, futFopExchange, underlyingSecType, underlyingConId));
        return this;
    }
    /**
     * Returns the mapping of single letter codes to exchange names given the mapping identifier.
     *
     * @param reqId The id of the request.
     * @param bboExchange The mapping identifier received from on tickReqParams event.
     */
    reqSmartComponents(reqId, bboExchange) {
        this.controller.schedule(() => this.controller.encoder.reqSmartComponents(reqId, bboExchange));
        return this;
    }
    /**
     * Requests pre-defined Soft Dollar Tiers.
     *
     * This is only supported for registered professional advisors and hedge and mutual funds who have configured Soft
     * Dollar Tiers in Account Management.
     *
     * Refer to:
     * https://www.interactivebrokers.com/en/software/am/am/manageaccount/requestsoftdollars.htm?Highlight=soft%20dollar%20tier.
     *
     * @param reqId The id of the request.
     */
    reqSoftDollarTiers(reqId) {
        this.controller.schedule(() => this.controller.encoder.reqSoftDollarTiers(reqId));
        return this;
    }
    /**
     * Requests tick-by-tick data.
     *
     * @param reqId Unique identifier of the request.
     * @param contract The [[Contract]] for which tick-by-tick data is requested.
     * @param tickType tick-by-tick data type: "Last", "AllLast", "BidAsk" or "MidPoint".
     * @param numberOfTicks number of ticks.
     * @param ignoreSize ignore size flag.
     */
    reqTickByTickData(reqId, contract, tickType, numberOfTicks, ignoreSize) {
        this.controller.schedule(() => this.controller.encoder.reqTickByTickData(reqId, contract, tickType, numberOfTicks, ignoreSize));
        return this;
    }
    /**
     * Requests the FA configuration A Financial Advisor can define three different configurations:
     *
     * - 1. Groups: offer traders a way to create a group of accounts and apply a single allocation method to all
     * accounts in the group.
     * - 2. Profiles: let you allocate shares on an account-by-account basis using a predefined calculation value.
     * - 3. Account Aliases: let you easily identify the accounts by meaningful names rather than account numbers.
     *
     * More information at
     * https://www.interactivebrokers.com/en/?f=%2Fen%2Fsoftware%2Fpdfhighlights%2FPDF-AdvisorAllocations.php%3Fib_entity%3Dllc
     *
     * @param faDataType The configuration to change. Set to 1, 2 or 3 as defined above.
     */
    requestFA(faDataType) {
        this.controller.schedule(() => this.controller.encoder.requestFA(faDataType));
        return this;
    }
    /**
     * Integrates API client and TWS window grouping.
     *
     * @param reqId The Id chosen for this subscription request.
     * @param groupId The display group for integration.
     */
    subscribeToGroupEvents(reqId, groupId) {
        this.controller.schedule(() => this.controller.encoder.subscribeToGroupEvents(reqId, groupId));
        return this;
    }
    /**
     * Updates the contract displayed in a TWS Window Group.
     *
     * @param reqId The ID chosen for this request.
     * @param contractInfo An encoded value designating a unique IB contract.
     * Possible values include:
     * - none = empty selection
     * - contractID = any non-combination contract. Examples 8314 for IBM SMART; 8314 for IBM ARCA
     * - combo = if any combo is selected Note: This request from the API does not get a TWS response unless an error
     *   occurs.
     */
    updateDisplayGroup(reqId, contractInfo) {
        this.controller.schedule(() => this.controller.encoder.updateDisplayGroup(reqId, contractInfo));
        return this;
    }
    /**
     * Cancels a TWS Window Group subscription.
     *
     * @param reqId The request ID.
     *
     * @sse [[subscribeToGroupEvents]]
     */
    unsubscribeFromGroupEvents(reqId) {
        this.controller.schedule(() => this.controller.encoder.unsubscribeToGroupEvents(reqId));
        return this;
    }
    /**
     * Changes the TWS/GW log level.
     *
     * The default is [[LOG_LEVEL.ERROR]]
     *
     * @param logLevel The log level.
     */
    setServerLogLevel(logLevel) {
        this.controller.schedule(() => this.controller.encoder.setServerLogLevel(logLevel));
        return this;
    }
    /**
     * Requests the user info of the logged user.
     */
    reqUserInfo(reqId) {
        this.controller.schedule(() => this.controller.encoder.reqUserInfo(reqId));
        return this;
    }
}
exports.IBApi = IBApi;
//# sourceMappingURL=api.js.map