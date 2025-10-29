"use strict";
/**
 * @jest-environment ./src/tests/nodb-test-environment
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const configuration_1 = __importDefault(require("../../common/configuration"));
describe("configuration", () => {
    test("ENV vars take priority", async () => {
        expect(configuration_1.default.env_config_test).toEqual("ENV");
    });
    test("ENV vars must be specified in includes array", async () => {
        expect(configuration_1.default.env_not_included_config_test).toEqual("default");
    });
    test("environment specific files take priority over local", async () => {
        const env = configuration_1.default.ci ? "development" : "test";
        expect(configuration_1.default.environment_config_test).toEqual(env);
    });
    test("local takes priority over default", async () => {
        if (!configuration_1.default.ci) {
            expect(configuration_1.default.local_config_test).toEqual("local");
        }
    });
    test("default is lowest priority", async () => {
        expect(configuration_1.default.default_config_test).toEqual("default");
    });
});
//# sourceMappingURL=configuration.test.js.map