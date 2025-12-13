"use strict";
/**
 * This App will print IBKR account modify orders to console.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../");
const configuration_1 = __importDefault(require("../common/configuration"));
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Modify order.";
const USAGE_TEXT = "Usage: modify-orders.js <options>";
const OPTION_ARGUMENTS = [
    ["price=<number>", "price of an order."],
    ["quantity=<number>", "Quantity of an order."],
];
const EXAMPLE_TEXT = "modify-orders.js -price=120 -quantity=10 -clientId=0 -orderId=2";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class ModifyOrdersApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        const id = +this.cmdLineArgs.orderId;
        const contract = {
            symbol: "AAPL",
            exchange: "SMART",
            currency: "USD",
            secType: __1.SecType.STK,
        };
        const order = {
            orderType: __1.OrderType.LMT,
            action: __1.OrderAction.BUY,
            lmtPrice: +this.cmdLineArgs.price,
            orderId: id,
            totalQuantity: +this.cmdLineArgs.quantity,
            account: configuration_1.default.ib_test_account,
            transmit: true,
        };
        this.api.modifyOrder(id, contract, order);
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.exit();
    }
}
// run the app
new ModifyOrdersApp().start();
//# sourceMappingURL=modify-order.js.map