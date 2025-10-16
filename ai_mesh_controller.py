import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(r'C:\ai_project_hub')
SHARED = ROOT / 'store' / 'ai_shared'
SHARED.mkdir(parents=True, exist_ok=True)

def run_py(script_path: Path, args=None, name='job'):
    args = args or []
    cmd = [sys.executable, str(script_path)] + args
    print(f'\n=== [{name}] {datetime.now().isoformat()} ===')
    print('> ' + ' '.join(cmd))

    # Force UTF-8 for the child process so emojis/special chars won’t crash on Windows
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if out.stdout:
            print(out.stdout.rstrip())
        if out.stderr:
            print('[stderr]\n' + out.stderr.rstrip())
        return out.returncode
    except Exception as e:
        print(f'!! {name} failed: {e}')
        return 1

def main():
    validator = ROOT / 'ai_validator_connect.py'
    trainer   = ROOT / 'ai_trainer_connect.py'

    if not validator.exists():
        print(f'!! Missing: {validator}')
        return 1
    if not trainer.exists():
        print(f'!! Missing: {trainer}')
        return 1

    rc1 = run_py(validator, name='validator')
    rc2 = run_py(trainer,   name='trainer')

    hb = SHARED / 'mesh_heartbeat.json'
    hb.write_text(
        '{"last_run":"%s","validator_rc":%d,"trainer_rc":%d}' %
        (datetime.now().isoformat(), rc1, rc2),
        encoding='utf-8'
    )
    print(f'\nWrote heartbeat: {hb}')

    if rc1 == 0 and rc2 == 0:
        print('\nALL GOOD (validator & trainer ran successfully)')
        return 0
    else:
        print('\nSOME CHECKS FAILED (see above logs and shared reports)')
        return max(rc1, rc2)

if __name__ == '__main__':
    sys.exit(main())
