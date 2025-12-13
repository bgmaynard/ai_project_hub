"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * This file implement test code for the reqAllOpenOrders function and openOrder event.
 */
const __1 = require("../../../..");
const configuration_1 = __importDefault(require("../../../../common/configuration"));
describe("RequestAllOpenOrders", () => {
    jest.setTimeout(10 * 1000);
    let ib;
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId,
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
    it("Test reqAllOpenOrders", (done) => {
        ib.on(__1.EventName.openOrder, (orderId, contract, order, orderState) => {
            // logger.info("openOrder message received");
            // todo add proper verification code here
            // expect(orderId).toBeTruthy(); We sometimes get zeros
        })
            .on(__1.EventName.openOrderEnd, () => {
            if (ib)
                ib.disconnect();
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqAllOpenOrders();
    });
});
//# sourceMappingURL=reqAllOpenOrders.test.js.map