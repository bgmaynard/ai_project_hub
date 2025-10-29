"use strict";
/**
 * This App will print the user info of the logged user.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the user info of the logged user.";
const USAGE_TEXT = "Usage: user-info.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "user-info.js -port=4002";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class App extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.api
            .getUserInfo()
            .then((whiteBrandingId) => {
            this.printText(`User Info. WhiteBrandingId: '${whiteBrandingId}'`);
            this.stop();
        })
            .catch((err) => {
            this.error(`getUserInfo failed with '${err.error.message}'`);
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
new App().start();
//# sourceMappingURL=user-info.js.map