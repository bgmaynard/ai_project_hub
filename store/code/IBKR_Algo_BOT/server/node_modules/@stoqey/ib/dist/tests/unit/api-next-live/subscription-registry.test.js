"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const logger_1 = __importDefault(require("../../../common/logger"));
describe("Subscription registry Tests", () => {
    jest.setTimeout(2_000);
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    let subscription$;
    let api;
    let error$;
    beforeEach(() => {
        api = new __1.IBApiNext();
        if (!error$) {
            error$ = api.errorSubject.subscribe((error) => {
                if (error.reqId === -1) {
                    logger_1.default.warn(`${error.error.message} (Error #${error.code})`);
                }
                else {
                    logger_1.default.error(`${error.error.message} (Error #${error.code}) ${error.advancedOrderReject ? error.advancedOrderReject : ""}`);
                }
            });
        }
        try {
            api.connect(clientId);
        }
        catch (error) {
            logger_1.default.error(error.message);
        }
    });
    afterEach(() => {
        if (api) {
            api.disconnect();
            api = undefined;
        }
    });
    it("Twice the same event callback bug", (done) => {
        // Two active subscriptions for the same Event issue #193
        subscription$ = api.getOpenOrders().subscribe({
            next: (_data) => {
                // console.log(data);
            },
            complete: () => {
                logger_1.default.info("getOpenOrders completed.");
                done();
            },
            error: (err) => {
                logger_1.default.error(`getOpenOrders failed with '${err.error.message}'`);
            },
        });
        api
            .getAllOpenOrders()
            .then((orders) => {
            logger_1.default.info(orders);
            subscription$.unsubscribe();
        })
            .catch((err) => {
            logger_1.default.error(`getAllOpenOrders failed with '${err}'`);
        });
    });
});
//# sourceMappingURL=subscription-registry.test.js.map