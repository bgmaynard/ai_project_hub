"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print daily PnL and unrealized PnL for a given account id.";
const USAGE_TEXT = "Usage: pnl.js <options>";
const OPTION_ARGUMENTS = [
    ["account", "(required) The IBKR account id."],
];
const EXAMPLE_TEXT = "pnl.js -account=DU1234567 -watch";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintPositionsApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        if (!this.cmdLineArgs.account) {
            this.error("-account argument missing.");
        }
        this.subscription$ = this.api
            .getPnL(this.cmdLineArgs.account, this.cmdLineArgs.model)
            .subscribe({
            next: (pnl) => {
                this.printObject(pnl);
                if (!this.cmdLineArgs.watch) {
                    this.stop();
                }
            },
            error: (err) => {
                this.error(`getPnL failed with '${err.error.message}'`);
            },
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.subscription$?.unsubscribe();
        this.exit();
    }
}
// run the app
new PrintPositionsApp().start();
//# sourceMappingURL=pnl.js.map