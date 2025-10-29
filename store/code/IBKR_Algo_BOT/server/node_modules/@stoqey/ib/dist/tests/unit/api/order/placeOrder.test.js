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
const orders_1 = require("../../sample-data/orders");
describe("Place Orders", () => {
    jest.setTimeout(20 * 1000);
    let ib;
    let clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId: clientId++, // increment clientId for each test so they don't interfere on each other
        });
    });
    afterEach(() => {
        if (ib) {
            ib.disconnect();
            ib = undefined;
        }
    });
    test("Stock placeOrder", (done) => {
        let refId;
        const refContract = contracts_1.sample_stock;
        const refOrder = orders_1.sample_order;
        let isSuccess = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, refContract, refOrder);
        })
            .on(__1.EventName.openOrder, (orderId, contract, order, _orderState) => {
            if (orderId == refId) {
                expect(contract.symbol).toEqual(refContract.symbol);
                expect(order.totalQuantity).toEqual(refOrder.totalQuantity);
            }
        })
            .on(__1.EventName.orderStatus, (orderId, _status, filled, remaining) => {
            if (!isSuccess && orderId == refId) {
                expect(filled).toEqual(0);
                expect(remaining).toEqual(refOrder.totalQuantity);
                isSuccess = true;
                ib.cancelOrder(orderId);
                done();
            }
        });
        ib.connect().on(__1.EventName.error, (error, code, reqId) => {
            if (reqId === __1.ErrorCode.NO_VALID_ID) {
                done(error.message);
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                if (error.message.includes("Warning:") ||
                    error.message.includes("Order Message:")) {
                    logger_1.default.warn(msg);
                }
                else {
                    ib.disconnect();
                    done(msg);
                }
            }
        });
    });
    test("What if Order", (done) => {
        let refId;
        const refContract = contracts_1.sample_etf;
        // const refOrder: Order = {
        //   orderType: OrderType.LMT,
        //   action: OrderAction.BUY,
        //   lmtPrice: 1,
        //   orderId: refId,
        //   totalQuantity: 2,
        //   tif: TimeInForce.DAY,
        //   transmit: true,
        //   whatIf: true,
        // };
        const refOrder = {
            ...orders_1.sample_order,
            goodAfterTime: undefined,
            whatIf: true,
        };
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, refContract, refOrder);
        }).on(__1.EventName.openOrder, (orderId, contract, order, orderState) => {
            if (orderId == refId) {
                expect(contract.symbol).toEqual(refContract.symbol);
                expect(order.totalQuantity).toEqual(refOrder.totalQuantity);
                if (orderState.minCommission || orderState.maxCommission) {
                    expect(orderState.commissionCurrency).toEqual(refContract.currency);
                    done();
                }
            }
        });
        ib.connect().on(__1.EventName.error, (error, code, reqId) => {
            if (reqId === __1.ErrorCode.NO_VALID_ID) {
                done(error.message);
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                if (error.message.includes("Warning:") ||
                    error.message.includes("Order Message:")) {
                    logger_1.default.warn(msg);
                }
                else {
                    ib.disconnect();
                    done(msg);
                }
            }
        });
    });
    test("Crypto placeOrder", (done) => {
        let refId;
        const refContract = contracts_1.sample_crypto;
        const refOrder = orders_1.sample_order;
        let isSuccess = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, refContract, refOrder);
        })
            .on(__1.EventName.openOrder, (orderId, contract, order, _orderState) => {
            if (orderId == refId) {
                expect(contract.symbol).toEqual(refContract.symbol);
                expect(order.totalQuantity).toEqual(refOrder.totalQuantity);
            }
        })
            .on(__1.EventName.orderStatus, (orderId, _status, filled, remaining) => {
            if (!isSuccess && orderId == refId) {
                expect(filled).toEqual(0);
                expect(remaining).toEqual(refOrder.totalQuantity);
                isSuccess = true;
                ib.cancelOrder(orderId);
                done();
            }
        });
        ib.connect().on(__1.EventName.error, (error, code, reqId) => {
            if (reqId === __1.ErrorCode.NO_VALID_ID) {
                done(error.message);
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                if (error.message.includes("Warning:") ||
                    error.message.includes("Order Message:")) {
                    logger_1.default.warn(msg);
                }
                else {
                    ib.disconnect();
                    done(msg);
                }
            }
        });
    });
    test("Option placeOrder", (done) => {
        let refId;
        const refContract = contracts_1.sample_option;
        const refOrder = orders_1.sample_order;
        let isSuccess = false;
        ib.once(__1.EventName.nextValidId, (orderId) => {
            refId = orderId;
            ib.reqOpenOrders().placeOrder(refId, refContract, refOrder);
        })
            .on(__1.EventName.openOrder, (orderId, contract, order, _orderState) => {
            if (orderId == refId) {
                expect(contract.symbol).toEqual(refContract.symbol);
                expect(order.totalQuantity).toEqual(refOrder.totalQuantity);
            }
        })
            .on(__1.EventName.orderStatus, (orderId, _status, filled, remaining) => {
            if (!isSuccess && orderId == refId) {
                expect(filled).toEqual(0);
                expect(remaining).toEqual(refOrder.totalQuantity);
                isSuccess = true;
                ib.cancelOrder(orderId);
                done();
            }
        });
        ib.connect().on(__1.EventName.error, (error, code, reqId) => {
            if (reqId === __1.ErrorCode.NO_VALID_ID) {
                done(error.message);
            }
            else {
                const msg = `[${reqId}] ${error.message} (Error #${code})`;
                if (error.message.includes("Warning:") ||
                    error.message.includes("Order Message:")) {
                    logger_1.default.warn(msg);
                }
                else {
                    ib.disconnect();
                    done(msg);
                }
            }
        });
    });
});
//# sourceMappingURL=placeOrder.test.js.map