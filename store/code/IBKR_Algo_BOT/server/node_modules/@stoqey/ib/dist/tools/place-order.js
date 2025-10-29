"use strict";
/**
 * This App will place orders to IBKR.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const path_1 = __importDefault(require("path"));
const __1 = require("../");
const configuration_1 = __importDefault(require("../common/configuration"));
const logger_1 = __importDefault(require("../common/logger"));
const ib_api_next_app_1 = require("./common/ib-api-next-app");
const scriptName = path_1.default.basename(__filename);
// The help text.
const DESCRIPTION_TEXT = "Place order.";
const USAGE_TEXT = `Usage: ${scriptName} <options>`;
const OPTION_ARGUMENTS = [
    ...ib_api_next_app_1.IBApiNextApp.DEFAULT_CONTRACT_OPTIONS,
    ["price=<number>", "price of an order."],
    ["quantity=<number>", "Quantity of an order."],
];
const EXAMPLE_TEXT = `${scriptName} -price=120 -symbol=AMZN -quantity=10`;
const awaitTimeout = (delay) => new Promise((resolve) => setTimeout(resolve, delay * 1000));
class App extends ib_api_next_app_1.IBApiNextApp {
    /**
     * Initialise the app.
     */
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.api
            .getNextValidOrderId()
            .then((id) => {
            const contract = {
                symbol: this.cmdLineArgs.symbol,
                exchange: this.cmdLineArgs.exchange,
                currency: this.cmdLineArgs.currency,
                secType: this.cmdLineArgs.sectype,
            };
            // this.printObject(contract);
            const order = {
                orderType: __1.OrderType.LMT,
                action: __1.OrderAction.BUY,
                lmtPrice: +this.cmdLineArgs.price,
                orderId: id,
                totalQuantity: +this.cmdLineArgs.quantity,
                account: configuration_1.default.ib_test_account,
                transmit: true,
            };
            this.api.placeOrder(id, contract, order);
            this.printText(`Order Id ${id} sent`);
            this.stop();
        })
            .catch((err) => {
            this.error(`getNextValidOrderId failed with '${err.error.message}'`);
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        // Give a 2 secs chance to get any server feedback before exiting
        awaitTimeout(2).then(() => {
            logger_1.default.info(`${scriptName} script done.`);
            this.exit();
        });
    }
}
// run the app
new App().start();
//# sourceMappingURL=place-order.js.map