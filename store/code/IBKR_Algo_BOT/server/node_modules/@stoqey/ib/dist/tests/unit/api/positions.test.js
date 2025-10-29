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
const logger_1 = __importDefault(require("../../../common/logger"));
describe("IBApi Tests", () => {
    jest.setTimeout(15_000);
    let ib;
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new __1.IBApi({
            host: configuration_1.default.ib_host,
            port: configuration_1.default.ib_port,
            clientId,
        });
    });
    afterEach(() => {
        if (ib) {
            ib.disconnect();
            ib = undefined;
        }
    });
    async function _delay(secs) {
        const res = await new Promise((resolve, _reject) => {
            setTimeout(() => {
                return resolve(true);
            }, secs * 1_000);
        });
        return res;
    }
    it("Test reqPositions / cancelPositions", (done) => {
        let positionsCount = 0;
        ib.on(__1.EventName.error, (err, code, id) => {
            expect(`${err.message} - code: ${code} - id: ${id}`).toBeFalsy();
        })
            .on(__1.EventName.position, (account, contract, pos, avgCost) => {
            expect(account).toBeTruthy();
            expect(contract).toBeTruthy();
            // expect(pos).toBeTruthy();  pos can be 0 when it has been closed today
            if (pos)
                expect(avgCost).toBeTruthy();
            positionsCount++;
        })
            .on(__1.EventName.positionEnd, () => {
            if (positionsCount) {
                ib.disconnect();
                done();
            }
            else {
                logger_1.default.error("No Positions received");
            }
        });
        ib.connect().reqPositions();
    });
    it("Test reqPositionsMulti / cancelPositionsMulti", (done) => {
        let refId = 45;
        let count = 0;
        ib.once(__1.EventName.connected, () => {
            ib.reqPositionsMulti(refId, "", "");
        })
            .on(__1.EventName.positionMulti, (reqId, account, modelCode, contract, _pos, _avgCost) => {
            //   console.log(
            //     "positionMulti",
            //     reqId,
            //     account,
            //     modelCode,
            //     JSON.stringify(contract),
            //     pos,
            //     avgCost,
            //   );
            expect(account).toBeTruthy();
            expect(contract).toBeTruthy();
        })
            .on(__1.EventName.positionMultiEnd, (reqId) => {
            count += 1;
            // console.log("positionMultiEnd", reqId);
            refId = reqId + 1;
            ib.cancelPositionsMulti(refId);
            // console.log("cancelPositionsMulti sent", refId);
            if (count < 3) {
                //   console.log("count", count);
                refId = refId + 1;
                ib.reqPositionsMulti(refId, "", "");
                // console.log("reqPositionsMulti sent", refId);
            }
            else {
                done();
            }
        });
        ib.on(__1.EventName.disconnected, () => done())
            .on(__1.EventName.error, (error, code, reqId) => {
            const msg = `[${reqId}] ${error.message} (Error #${code})`;
            (0, __1.isNonFatalError)(code, error) ? logger_1.default.warn(msg) : logger_1.default.error(msg);
        })
            .connect();
    });
});
//# sourceMappingURL=positions.test.js.map