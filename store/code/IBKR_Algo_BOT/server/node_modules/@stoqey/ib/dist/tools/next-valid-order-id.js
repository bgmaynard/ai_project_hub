"use strict";
/**
 * This App will print the next valid unused order id.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// Help text and command line parsing                                          //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the next valid unused order id.";
const USAGE_TEXT = "Usage: next-valid-order-id.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "next-valid-order-id.js -host=localhost -port=4002";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintNextUnusedOrderIdApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        // print next unused order id
        this.api
            .getNextValidOrderId()
            .then((id) => {
            this.printText(`${id}`);
            this.exit();
        })
            .catch((err) => {
            this.error(`getNextValidOrderId failed with '${err.error.message}'`);
        });
    }
}
// run the app
new PrintNextUnusedOrderIdApp().start();
//# sourceMappingURL=next-valid-order-id.js.map