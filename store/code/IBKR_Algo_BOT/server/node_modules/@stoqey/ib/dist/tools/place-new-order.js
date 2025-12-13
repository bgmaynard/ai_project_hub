"use strict";
/**
 * This App will print IBKR account place new orders to console.
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
const DESCRIPTION_TEXT = "Place new order.";
const USAGE_TEXT = "Usage: place-new-orders.js <options>";
const OPTION_ARGUMENTS = [
    ["price=<number>", "price of an order."],
    ["symbol=<name>", "The symbol name."],
    ["quantity=<number>", "Quantity of an order."],
];
const EXAMPLE_TEXT = "place-new-orders.js -price=120 -symbol=AMZN -quantity=10 -clientId=0";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PlaceNewOrdersApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        const contract = {
            symbol: this.cmdLineArgs.symbol,
            exchange: "SMART",
            currency: "USD",
            secType: __1.SecType.STK,
        };
        const order = {
            orderType: __1.OrderType.LMT,
            action: __1.OrderAction.BUY,
            lmtPrice: +this.cmdLineArgs.price,
            totalQuantity: +this.cmdLineArgs.quantity,
            account: configuration_1.default.ib_test_account,
            transmit: true,
        };
        this.api.placeNewOrder(contract, order).then((orderId) => {
            this.printText(orderId.toString());
            this.stop();
        });
        //setTimeout(process.exit(0), 3000);
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.exit();
    }
}
// run the app
new PlaceNewOrdersApp().start();
//# sourceMappingURL=place-new-order.js.map