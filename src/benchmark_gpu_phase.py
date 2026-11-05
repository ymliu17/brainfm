"""Explicit GPU-to-CPU handoff at audited final GPU-use boundaries."""
import fcntl
import json
import os
from pathlib import Path
import select
import sys
import time
from contextlib import contextmanager

FD_ENV = 'BENCHMARK_GPU_LEASE_FD'
PHASE_ENV = 'BENCHMARK_GPU_PHASE_FILE'
LOCK_ENV = 'BENCHMARK_GPU_LOCK_PATH'
STAGED_SCRIPTS = frozenset({
    'mci_axis_transfer.py', 'rct_probe.py', 'rct_loso.py',
    'nacc_probe_adapter.py', 'probe_readouts.py', 'learning_curve.py',
    'fusion_increment.py', 'rct_clinical_increment.py', 'rct_hrv_trait.py',
    'rct_inscanner_hr.py', 'nacc_posttransfer.py', 'transfer_calibration.py',
    'benchmark_extension.py', 'hrv_ppg_benchmark.py',
    'benchmark_adapted_fingerprint.py', 'fingerprint_adni_matched.py',
})
TRAINING_SCRIPTS = frozenset({'nacc_posttrain.py', 'benchmark_training_controls.py'})


def lease_mode(command):
    name = Path(command[1]).name
    return 'staged' if name in STAGED_SCRIPTS else ('training' if name in TRAINING_SCRIPTS else 'whole_task')


def _phase(phase):
    marker = os.environ.get(PHASE_ENV)
    if marker:
        p = Path(marker); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        d = dict(pid=os.getpid(), phase=phase, updated_at=time.time())
        if phase == 'cpu': d['gpu_released_at'] = d['updated_at']
        tmp.write_text(json.dumps(d)); tmp.replace(p)


def acquire_gpu_for_stage():
    """Block before GPU work, not during preceding CPU analysis; safely reacquire."""
    if FD_ENV in os.environ:
        return False  # Already owned, including a coordinator-inherited lease.
    path = os.environ.get(LOCK_ENV)
    if not path:
        return False  # Standalone execution keeps its existing behavior.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        _phase('waiting_gpu')
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.environ[FD_ENV] = str(fd)
        _phase('gpu')
    except BaseException:
        os.environ.pop(FD_ENV, None); os.close(fd); raise
    return True


@contextmanager
def gpu_stage():
    """Lease a short GPU block; nested scopes leave the outer lease intact."""
    acquired = acquire_gpu_for_stage()
    try:
        yield
    finally:
        if acquired:
            release_gpu_for_cpu_tail()


def gpu_call(function, *args, **kwargs):
    with gpu_stage():
        return function(*args, **kwargs)


def has_future_gpu_work(kinds):
    """Only audited FC/amplitude/Tucker families are known to be CPU-only."""
    return any(kind not in {'fc', 'amp', 'tucker'} for kind in kinds)


def release_gpu_for_cpu_tail():
    """Release the inherited shared lease only after the final GPU operation.

    Closing a descriptor alone would leave the parent's copy locked. LOCK_UN
    acts on their shared open-file description. The caller must never use the
    GPU again without reacquiring; this is not an idle-utilization test.
    Standalone runs without a coordinator lease are unchanged.
    """
    value = os.environ.get(FD_ENV)
    if value is None:
        return False
    fd = int(value)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    os.environ.pop(FD_ENV, None)
    _phase('cpu')
    print('GPU phase complete; shared lease released for CPU evaluation', flush=True)
    return True


class DetachedFusionCPU:
    """Monitor an audited fusion or RCT-probe CPU tail without reruns."""
    def __init__(self, job):
        self.script = Path(job['command'][1]).name
        if self.script not in {'fusion_increment.py', 'rct_probe.py'} or job.get('phase') != 'cpu':
            raise ValueError('Only an explicitly audited fusion/RCT-probe CPU tail may be adopted')
        if self.script == 'rct_probe.py' and not job.get('handoff_expected_args'):
            raise ValueError('RCT handoff requires the audited original argument values')
        self.job = job
        self.pid = job['pid']
        self.fd = None
        try:
            self.fd = os.pidfd_open(self.pid)
            if not select.select([self.fd], [], [], 0)[0]:
                actual = Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')[:-1]
                if actual != [s.encode() for s in job['command']]:
                    raise ValueError('Fusion handoff process identity mismatch')
        except ProcessLookupError:
            pass

    def poll(self):
        if self.fd is not None:
            if not select.select([self.fd], [], [], 0)[0]:
                return None
            os.close(self.fd)
            self.fd = None
        args = self.job['command']
        p = Path(args[args.index('--out') + 1])
        valid = p.exists() and p.stat().st_mtime >= self.job['started']
        if valid:
            try:
                d = json.loads(p.read_text())
                if self.script == 'fusion_increment.py':
                    valid = (d['args']['tag'] == args[args.index('--tag') + 1]
                             and len(d['scores']) == 9 and len(d['increments']) == 9
                             and set(d['scores']) == set(d['oof_predictions'])
                             and all(len(v) == len(d['y']) == len(d['subjects'])
                                     for v in d['oof_predictions'].values()))
                else:
                    expected = self.job['handoff_expected_args']
                    cells = {(k, t, c) for k in expected['kinds'] for t in expected['targets'] for c in expected['cohorts']}
                    valid = (all(d['args'][k] == v for k, v in expected.items())
                             and len(d['rows']) == len(cells)
                             and {(r['kind'], r['target'], r['cohort']) for r in d['rows']} == cells
                             and all('score' in r and 'p_perm' in r for r in d['rows'])
                             and d['cross_cohort'] == [])
            except (ValueError, KeyError, TypeError, OSError):
                valid = False
        self.job['completion_verification'] = 'Fresh complete evaluation result after original worker termination; original exit status unavailable'
        self.job['completion_artifact'] = str(p)
        return 0 if valid else 1
