"use strict";
/**
 * This App will print all positions on your IBKR accounts to console.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints all positions on your IBKR accounts to console.";
const USAGE_TEXT = "Usage: positions.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "positions.js -watch";
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
        this.subscription$ = this.api.getPositions().subscribe({
            next: (positions) => {
                this.printObject(positions);
                if (!this.cmdLineArgs.watch) {
                    this.stop();
                }
            },
            error: (err) => {
                this.error(`getPositions failed with '${err.error.message}'`);
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
//# sourceMappingURL=positions.js.map