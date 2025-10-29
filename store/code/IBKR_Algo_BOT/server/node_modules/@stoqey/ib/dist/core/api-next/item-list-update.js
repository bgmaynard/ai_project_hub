"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextItemListUpdate = void 0;
/**
 * @internal
 *
 * Implementation for the DataUpdate interface.
 */
class IBApiNextItemListUpdate {
    constructor(all, added, changed, removed) {
        this.all = all;
        this.added = added;
        this.changed = changed;
        this.removed = removed;
    }
}
exports.IBApiNextItemListUpdate = IBApiNextItemListUpdate;
//# sourceMappingURL=item-list-update.js.map