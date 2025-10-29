"use strict";
/**
 * This App will print real-time updates of the IBKR account open orders.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the account open orders.";
const USAGE_TEXT = "Usage: open-orders-updates.js <options>";
const OPTION_ARGUMENTS = [["bind", "auto bind orders"]];
const EXAMPLE_TEXT = "open-orders-updates.js";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class OpenOrdersApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.subscription$ = this.api.getOpenOrders().subscribe({
            next: (data) => {
                this.printObject(data);
            },
            error: (err) => {
                this.error(`getOpenOrders failed with '${err.error.message}'`);
            },
            complete: () => {
                console.log("getOpenOrders completed.");
            },
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        console.log("app stopping.");
        this.subscription$?.unsubscribe();
        this.exit();
    }
}
// run the app
new OpenOrdersApp().start();
//# sourceMappingURL=open-orders.js.map