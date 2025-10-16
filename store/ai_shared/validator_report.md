# AI Validator Report
- Generated: 2025-10-14 22:46:45
- Files scanned: 52

## Missing Python modules (import not found)
- `` (pip install )
- `App` (pip install App)
- `EASY_MTF_TRAINER_V2` (pip install EASY_MTF_TRAINER_V2)
- `IBKR_Algo_BOT` (pip install IBKR_Algo_BOT)
- `ReactDOM` (pip install ReactDOM)
- `broker_if` (pip install broker_if)
- `improved_backtest` (pip install improved_backtest)
- `lstm_model_complete` (pip install lstm_model_complete)
- `lstm_training_pipeline` (pip install lstm_training_pipeline)
- `mtf_features_simple` (pip install mtf_features_simple)
- `react` (pip install react)

## Pytest (summary)
- returncode: 3
```
               ~~~~~~~~~~~~^^
INTERNALERROR>   File "C:\Users\bgmay\AppData\Local\Programs\Python\Python313\Lib\site-packages\_pytest\python.py", line 551, in _getobj
INTERNALERROR>     return importtestmodule(self.path, self.config)
INTERNALERROR>   File "C:\Users\bgmay\AppData\Local\Programs\Python\Python313\Lib\site-packages\_pytest\python.py", line 498, in importtestmodule
INTERNALERROR>     mod = import_path(
INTERNALERROR>         path,
INTERNALERROR>     ...<2 lines>...
INTERNALERROR>         consider_namespace_packages=config.getini("consider_namespace_packages"),
INTERNALERROR>     )
INTERNALERROR>   File "C:\Users\bgmay\AppData\Local\Programs\Python\Python313\Lib\site-packages\_pytest\pathlib.py", line 587, in import_path
INTERNALERROR>     importlib.import_module(module_name)
INTERNALERROR>     ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
INTERNALERROR>   File "C:\Users\bgmay\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
INTERNALERROR>     return _bootstrap._gcd_import(name[level:], package, level)
INTERNALERROR>            ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
INTERNALERROR>   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
INTERNALERROR>   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
INTERNALERROR>   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
INTERNALERROR>   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
INTERNALERROR>   File "C:\Users\bgmay\AppData\Local\Programs\Python\Python313\Lib\site-packages\_pytest\assertion\rewrite.py", line 186, in exec_module
INTERNALERROR>     exec(co, module.__dict__)
INTERNALERROR>     ~~~~^^^^^^^^^^^^^^^^^^^^^
INTERNALERROR>   File "C:\ai_project_hub\store\code\IBKR_Algo_BOT\tests_integration\test_lstm.py", line 20, in <module>
INTERNALERROR>     exit(1)
INTERNALERROR>     ~~~~^^^
INTERNALERROR>   File "<frozen _sitebuiltins>", line 26, in __call__
INTERNALERROR> SystemExit: 1


```

## IBKR API Probe
- ✅ `/api/ibkr/test` reachable
```
{
  "client_exists": true,
  "connected": true,
  "ibkr_available": true,
  "wrapper_exists": true
}
```