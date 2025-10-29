"use strict";
/**
 * This App will print the accounts to which the logged user has access to.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the accounts to which the logged user has access to.";
const USAGE_TEXT = "Usage: managed-accts.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "managed-accts.js -port=4002";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintManagedAcctsApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.api
            .getManagedAccounts()
            .then((accounts) => {
            this.printObject(accounts);
            this.stop();
        })
            .catch((err) => {
            this.error(`getManagedAccounts failed with '${err.error.message}'`);
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
new PrintManagedAcctsApp().start();
//# sourceMappingURL=managed-accts.js.map