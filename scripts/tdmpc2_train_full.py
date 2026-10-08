#!/usr/bin/env python
"""Launch bounded nominal pilot or gated full training; capture subprocess evidence."""
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fluidgym_rl.pilot import parser, blockers, hydra_overrides, wandb_environment
from fluidgym_rl.report import collect_metadata
from fluidgym_rl.upstream import make_tdmpc2_workcopy, repo_root, _changed_files


def main():
    a = parser().parse_args()
    errors = blockers(a)
    if errors:
        print('TRAINING BLOCKED:\n  - ' + '\n  - '.join(errors))
        return 3
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = Path(a.out or Path(os.environ.get('FGRL_RESULTS_DIR', 'results')) / ('tdmpc2_' + a.mode) / run_id).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'status.json').write_text(json.dumps({'status': 'initializing', 'run_id': run_id}))
    try:
        work = (Path(a.work_dir or os.environ.get('FGRL_WORK_DIR', tempfile.gettempdir())) / ('tdmpc2_' + run_id)).resolve()
        wc = make_tdmpc2_workcopy(work)
        patch = repo_root() / 'patches/tdmpc2_logging.patch'
        # Apply only to the disposable copy; no changes to the pinned clone.
        subprocess.run(['git', 'init', '-q', str(work)], check=True)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=work, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=work, check=True)
        changed = _changed_files(Path(wc['upstream']['path']), work)
        expected = sorted(wc['changed_files'] + ['tdmpc2/common/logger.py'])
        if changed != expected:
            raise RuntimeError(f'unexpected training workcopy modifications: {changed}')
        wc['changed_files'] = changed
        import hashlib
        wc['logging_patch_sha256'] = hashlib.sha256(patch.read_bytes()).hexdigest()
        overrides = hydra_overrides(a, run_id)
        spec = {'args': vars(a), 'run_id': run_id, 'out': str(out), 'workcopy': wc,
                'overrides': overrides, 'metadata': collect_metadata(),
                'checkpoint_semantics': 'model weights only; not exact training resume'}
        (out / 'launch.json').write_text(json.dumps(spec, indent=2, default=str))
        command = [sys.executable, str(repo_root() / 'scripts/tdmpc2_pilot_worker.py'), str(out / 'launch.json')]
        environment = {**os.environ, **wandb_environment(a), 'PYTHONUNBUFFERED': '1',
                       'PYTHONPATH': os.pathsep.join([str(repo_root() / 'src'), wc['code_dir'], os.environ.get('PYTHONPATH', '')])}
        print('Run directory:', out, flush=True)
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, cwd=out, env=environment, stdout=stdout, stderr=stderr)
        rc = result.returncode
        (out / 'exit_code.txt').write_text(str(rc))
        if rc:
            print((out / 'stderr.log').read_text(), file=sys.stderr)
        else:
            evidence = json.loads((out / 'status.json').read_text())
            if evidence['status'] != ('validated' if a.validate_only else 'completed'):
                raise RuntimeError('child exited without completion evidence')
        if rc:
            (out / 'status.json').write_text(json.dumps({'status': 'failed', 'exit_code': rc}))
        print('Exit code:', rc, '| evidence:', out)
        return rc if rc >= 0 else 128 - rc
    except BaseException as exc:
        (out / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}))
        raise


if __name__ == '__main__':
    sys.exit(main())
