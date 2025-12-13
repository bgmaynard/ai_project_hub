"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * This file implement test code for the public API interfaces.
 */
const __1 = require("../../..");
const configuration_1 = __importDefault(require("../../../common/configuration"));
describe("IBApi Historical data Tests", () => {
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
    it("Returns the correct head timestamp", (done) => {
        const referenceID = 42;
        ib.once(__1.EventName.connected, () => {
            const contract = new __1.Stock("AAPL");
            ib.reqHeadTimestamp(referenceID, contract, __1.WhatToShow.TRADES, true, 2)
                .on(__1.EventName.headTimestamp, (requestId, headTimestamp) => {
                if (requestId === referenceID) {
                    expect(headTimestamp).toEqual("345479400");
                    ib.disconnect();
                }
            })
                .on(__1.EventName.disconnected, done)
                .on(__1.EventName.error, (err, code, requestId) => {
                if (requestId === referenceID) {
                    done(`[${requestId}] ${err.message} (#${code})`);
                }
            });
        });
        ib.connect();
    });
});
//# sourceMappingURL=head-timestamp.test.js.map