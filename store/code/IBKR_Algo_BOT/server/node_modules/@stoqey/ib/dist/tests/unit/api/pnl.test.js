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
    jest.setTimeout(10000);
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
    let _account; // maintain account name for further tests
    let _conId; // maintain for conId for  further tests
    it("Test reqPositions / cancelPositions", (done) => {
        let positionsCount = 0;
        ib.on(__1.EventName.error, (err, code, id) => {
            expect(`${err.message} - code: ${code} - id: ${id}`).toBeFalsy();
        })
            .on(__1.EventName.position, (account, contract, pos, avgCost) => {
            if (_account === undefined) {
                _account = account;
            }
            if (_conId === undefined && pos) {
                _conId = contract.conId;
                // console.info(JSON.stringify(contract));
            }
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
    it("Test reqPnL / cancelPnL", (done) => {
        const refId = 43;
        let received = false;
        ib.on(__1.EventName.pnl, (reqId, _pnl) => {
            expect(reqId).toEqual(refId);
            // expect(pnl).toBeTruthy();
            if (!received) {
                ib.cancelPnL(reqId);
                ib.disconnect();
            }
            received = true;
        })
            .on(__1.EventName.disconnected, () => {
            done();
        })
            .on(__1.EventName.error, (err, code, reqId) => {
            if (reqId == refId)
                done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect().reqPnL(refId, _account);
    });
    it("Test reqPnLSingle / cancelPnLSingle", (done) => {
        const refId = 44;
        let received = false;
        ib.once(__1.EventName.connected, () => {
            // console.log("reqPnLSingle", refId);
            ib.reqPnLSingle(refId, _account, "", _conId);
        }).on(__1.EventName.pnlSingle, (reqId, pos, _dailyPnL, unrealizedPnL, _realizedPnL, value) => {
            // console.log(
            //   "pnlSingle",
            //   reqId,
            //   pos,
            //   _dailyPnL,
            //   unrealizedPnL,
            //   _realizedPnL,
            //   value,
            // );
            expect(reqId).toEqual(refId);
            expect(pos).toBeTruthy();
            // expect(dailyPnL).toBeTruthy(); We may have no daily PnL (on week-ends)
            expect(unrealizedPnL).toBeTruthy();
            // expect(realizedPnL).toBeTruthy();  We may have no realized PnL today
            expect(value).toBeTruthy();
            if (!received) {
                ib.cancelPnLSingle(reqId);
                ib.disconnect();
                done();
            }
            received = true;
        });
        ib.on(__1.EventName.disconnected, () => done())
            // .on(EventName.info, (msg, code) => console.info("INFO", code, msg))
            .on(__1.EventName.error, (err, code, reqId) => {
            const msg = `[${reqId}] ${err.message} (#${code})`;
            if (reqId > 0 &&
                code != __1.ErrorCode.INVALID_POSITION_TRADE_DERIVATED_VALUE) {
                done(msg);
            }
            else {
                logger_1.default.error(msg);
            }
        })
            .connect();
    });
});
//# sourceMappingURL=pnl.test.js.map