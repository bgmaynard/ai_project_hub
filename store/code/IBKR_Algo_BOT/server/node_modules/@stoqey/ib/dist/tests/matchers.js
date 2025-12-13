"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/* eslint-disable @typescript-eslint/no-explicit-any */
const ajv_1 = __importDefault(require("ajv"));
const ajv = new ajv_1.default();
expect.extend({
    toMatchSchema(data, schema) {
        const pass = ajv.validate(schema, data);
        if (pass) {
            return {
                message: () => "expected to not match schema",
                pass: true,
            };
        }
        return {
            message: () => {
                const errors = JSON.stringify(ajv.errors, null, 2);
                return `expected to match schema, instead ${errors}`;
            },
            pass: false,
        };
    },
});
//# sourceMappingURL=matchers.js.map