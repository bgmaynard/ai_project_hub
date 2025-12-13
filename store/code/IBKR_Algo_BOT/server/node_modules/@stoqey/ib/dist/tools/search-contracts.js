"use strict";
/**
 * This App will search contracts matching a given text pattern and print it
 * to console.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
///////////////////////////////////////////////////////////////////////////////
// The help text                                                             //
///////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Search contracts matching a given text pattern.";
const USAGE_TEXT = "Usage: search-contract.js <options>";
const OPTION_ARGUMENTS = [
    ["pattern=<text>", "The name or symbol to search."],
];
const EXAMPLE_TEXT = "search-contract.ts -text=Apple";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintContractSearchApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        if (!this.cmdLineArgs.pattern) {
            this.error("-pattern argument missing.");
        }
        this.api
            .getMatchingSymbols(this.cmdLineArgs.pattern)
            .then((searchResult) => {
            this.printObject(searchResult);
            this.stop();
        })
            .catch((err) => {
            this.error(`searchContracts failed with '${err.error.message}'`);
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.exit();
    }
}
// run the app
new PrintContractSearchApp().start();
//# sourceMappingURL=search-contracts.js.map