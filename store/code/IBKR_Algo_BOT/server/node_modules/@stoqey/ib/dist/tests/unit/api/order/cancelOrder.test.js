"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * This file implement test code for the placeOrder function .
 */
const __1 = require("../../../..");
const configuration_1 = __importDefault(require("../../../../common/configuration"));
const logger_1 = __importDefault(require("../../../../common/logger"));
const contracts_1 = require("../../sample-data/contracts");
describe("CancelOrder", () => {
    jest.setTimeout(20 * 1000);
    let ib;
    let clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    const contract = contracts_1.sample_etf;
    const order = {
        orderType: __1.OrderType.LMT,
        action: __1.OrderAction.BUY,
        lmtPrice: 3,
        totalQuantity: 1,
        tif: __1.TimeInForce.DAY,
        outsideRth: true,
        transmit: true,
        goodAfterTime: "20300101-01:01:01",
    };
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId: clientId++, // increment clientId for each test so they don't interfere on each other
        });
        // logger.info("IBApi created");
    });
    afterEach(() => {
        if (ib) {
            ib.disconnect();
            ib = undefined;
        }
        // logger.info("IBApi disconnected");
    });
    test("cancelOrder", (done) => {
        let refId;
        let isCancelling = false;
        let isDone = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, contract, order);
        })
            .on(__1.EventName.orderStatus, (orderId, status, _filled, _remaining, _avgFillPrice, _permId, _parentId, _lastFillPrice, _clientId, _whyHeld, _mktCapPrice) => {
            if (orderId === refId) {
                // console.log(orderId, status);
                if (isDone) {
                    // ignore any message
                }
                else if (!isCancelling) {
                    // [OrderStatus.PreSubmitted, OrderStatus.Submitted].includes(
                    //   status as OrderStatus,
                    // )
                    isCancelling = true;
                    ib.cancelOrder(orderId);
                }
                else {
                    if ([
                        __1.OrderStatus.PendingCancel,
                        __1.OrderStatus.ApiCancelled,
                        __1.OrderStatus.Cancelled,
                    ].includes(status)) {
                        isDone = true;
                        done();
                    }
                }
            }
        })
            .on(__1.EventName.error, (error, code, reqId) => {
            if (code == __1.ErrorCode.ORDER_CANCELLED &&
                reqId == refId &&
                isCancelling) {
                // Alright, we can safely ignore
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                isDone = true;
                done(msg);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error)
                ? logger_1.default.warn(msg)
                : isCancelling
                    ? logger_1.default.info(msg)
                    : logger_1.default.error(msg);
        })
            .connect();
    });
    test("cancelOrder immediate", (done) => {
        let refId;
        let isCancelling = false;
        let isDone = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, contract, order);
        })
            .on(__1.EventName.orderStatus, (orderId, status) => {
            // console.log(orderId, status, isCancelling, isDone);
            if (orderId === refId) {
                // console.log(orderId, status);
                if (isDone) {
                    // ignore any message
                }
                else if (!isCancelling) {
                    // [OrderStatus.PreSubmitted, OrderStatus.Submitted].includes(
                    //   status as OrderStatus,
                    // )
                    isCancelling = true;
                    ib.cancelOrder(orderId, "");
                }
                else {
                    if ([
                        __1.OrderStatus.PendingCancel,
                        __1.OrderStatus.ApiCancelled,
                        __1.OrderStatus.Cancelled,
                    ].includes(status)) {
                        isDone = true;
                        done();
                    }
                }
            }
        })
            .on(__1.EventName.error, (error, code, reqId) => {
            if (code == __1.ErrorCode.ORDER_CANCELLED &&
                reqId == refId &&
                isCancelling) {
                // Alright, we can safely ignore
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                isDone = true;
                done(msg);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error)
                ? logger_1.default.warn(msg)
                : isCancelling
                    ? logger_1.default.info(msg)
                    : logger_1.default.error(msg);
        })
            .connect();
    });
    test("cancelOrder later", (done) => {
        // NOTE: this test is not correctly written, but the API doesn't behave as *I* expected neither
        let refId;
        let isCancelling = false;
        let isDone = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, contract, order);
        })
            .on(__1.EventName.orderStatus, (orderId, status) => {
            // console.log(orderId, status, isCancelling, isDone);
            if (orderId === refId) {
                if (isDone) {
                    // ignore any message
                }
                else if (!isCancelling) {
                    isCancelling = true;
                    ib.cancelOrder(orderId, "20310101-23:59:59");
                }
                else if ([
                    __1.OrderStatus.PendingCancel,
                    __1.OrderStatus.ApiCancelled,
                    __1.OrderStatus.Cancelled,
                ].includes(status)) {
                    isDone = true;
                    done();
                }
            }
        })
            .on(__1.EventName.error, (error, code, reqId) => {
            if (isDone) {
                // ignore any message
            }
            else if (code == __1.ErrorCode.ORDER_CANCELLED &&
                reqId == refId &&
                isCancelling) {
                if (!isDone) {
                    isDone = true;
                    done();
                }
            }
            else if (!(0, __1.isNonFatalError)(code, error)) {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                isDone = true;
                done(msg);
            }
        });
        ib.on(__1.EventName.info, (msg, code) => logger_1.default.info(code, msg))
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error)
                ? logger_1.default.warn(msg)
                : isCancelling
                    ? logger_1.default.info(msg)
                    : logger_1.default.error(msg);
        })
            .connect();
    });
});
//# sourceMappingURL=cancelOrder.test.js.map