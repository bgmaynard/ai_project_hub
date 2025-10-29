"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const api_1 = require("../../../api/api");
const event_name_1 = require("../../../api/data/enum/event-name");
const configuration_1 = __importDefault(require("../../../common/configuration"));
describe("IBApi connection Tests", () => {
    jest.setTimeout(5 * 1000);
    let ib;
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    beforeEach(() => {
        ib = new api_1.IBApi({
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
    test("Connect", (done) => {
        // logger.info("Starting Connect");
        ib.on(event_name_1.EventName.connected, () => {
            done();
        }).on(event_name_1.EventName.error, (err, code, reqId) => {
            done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("Disconnect", (done) => {
        // logger.info("Starting Disconnect");
        ib.on(event_name_1.EventName.connected, () => {
            ib.disconnect();
            ib = undefined;
        })
            .on(event_name_1.EventName.disconnected, () => {
            done();
        })
            .on(event_name_1.EventName.error, (err, code, reqId) => {
            done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
    test("Connect / disconnect", (done) => {
        // logger.info("Starting (Dis)Connect");
        ib.on(event_name_1.EventName.connected, () => {
            ib.reqCurrentTime();
        })
            .on(event_name_1.EventName.currentTime, (time) => {
            expect(time).toBeTruthy();
            if (ib)
                ib.disconnect();
        })
            .on(event_name_1.EventName.disconnected, () => {
            done();
        })
            .on(event_name_1.EventName.error, (err, code, reqId) => {
            done(`[${reqId}] ${err.message} (#${code})`);
        });
        ib.connect();
    });
});
//# sourceMappingURL=connect.test.js.map