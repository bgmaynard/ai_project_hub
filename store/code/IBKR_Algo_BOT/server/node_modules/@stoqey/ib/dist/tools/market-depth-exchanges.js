"use strict";
/**
 * This App will print the venues for which market data is returned on getMarketDepthL2 (those with market makers)
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// Help text and command line parsing                                          //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the venues for which market data is returned on getMarketDepthL2 (those with market makers).";
const USAGE_TEXT = "Usage: market-depth-exchanges.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "market-depth-exchanges.js -host=localhost -port=4002";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintMarketDepthExchangesApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        // print current time
        this.api
            .getMarketDepthExchanges()
            .then((data) => {
            this.printObject(data);
            this.exit();
        })
            .catch((err) => {
            this.error(`getMarketDepthExchanges failed with '${err.error.message}'`);
        });
    }
}
// run the app
new PrintMarketDepthExchangesApp().start();
//# sourceMappingURL=market-depth-exchanges.js.map