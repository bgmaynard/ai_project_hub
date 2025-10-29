export interface Configuration {
    ci: string;
    env_config_test: string;
    env_not_included_config_test: string;
    default_config_test: string;
    local_config_test: string;
    environment_config_test: string;
    ib_host: string;
    ib_port: number;
    ib_test_account: string;
    default_client_id: number;
    client_version: number;
    max_req_per_second: number;
    environment: string;
    isProduction: boolean;
    isStaging: boolean;
    isDevelopment: boolean;
    isTest: boolean;
    isLocal: boolean;
}
declare let configuration: Configuration;
export declare function get(): Configuration;
export default configuration;
