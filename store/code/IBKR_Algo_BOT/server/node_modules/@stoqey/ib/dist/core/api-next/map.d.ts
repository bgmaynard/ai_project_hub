/**
 * ES6 Map class + some custom convenience functions.
 */
export declare class IBApiNextMap<K, V> extends Map<K, V> {
    getOrAdd(k: K, factory: (k: K) => V): V;
}
