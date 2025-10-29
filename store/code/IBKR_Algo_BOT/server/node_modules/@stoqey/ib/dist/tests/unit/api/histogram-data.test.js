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
describe("IBApi Histogram data Tests", () => {
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
    it("Histogram market data", (done) => {
        const referenceID = 46;
        ib.once(__1.EventName.connected, () => {
            const contract = new __1.Stock("AAPL");
            ib.reqHistogramData(referenceID, contract, true, 1, __1.DurationUnit.WEEK)
                .on(__1.EventName.histogramData, (requestID, data) => {
                if (requestID === referenceID) {
                    expect(requestID).toEqual(referenceID);
                    expect(data.length).toBeGreaterThan(0);
                    ib.disconnect();
                }
            })
                .on(__1.EventName.disconnected, done)
                .on(__1.EventName.error, (err, code, requestID) => {
                if (requestID == referenceID) {
                    done(`[${requestID}] ${err.message} (#${code})`);
                }
            });
        });
        ib.connect();
    });
});
//# sourceMappingURL=histogram-data.test.js.map