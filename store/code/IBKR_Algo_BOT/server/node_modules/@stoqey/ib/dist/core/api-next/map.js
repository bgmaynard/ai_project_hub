"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextMap = void 0;
/**
 * ES6 Map class + some custom convenience functions.
 */
class IBApiNextMap extends Map {
    getOrAdd(k, factory) {
        let r = this.get(k);
        if (r === undefined) {
            r = factory(k);
            this.set(k, r);
        }
        return r;
    }
}
exports.IBApiNextMap = IBApiNextMap;
//# sourceMappingURL=map.js.map