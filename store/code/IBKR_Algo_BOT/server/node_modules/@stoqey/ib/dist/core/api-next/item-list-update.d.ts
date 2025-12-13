import { ItemListUpdate } from "../../api-next/common/item-list-update";
/**
 * @internal
 *
 * Implementation for the DataUpdate interface.
 */
export declare class IBApiNextItemListUpdate<T> implements ItemListUpdate<T> {
    readonly all: T;
    readonly added?: T;
    readonly changed?: T;
    readonly removed?: T;
    constructor(all: T, added?: T, changed?: T, removed?: T);
}
