"""Process-inherited GPU lease shared by independent benchmark coordinators."""
from pathlib import Path
import fcntl
ROOT=Path(__file__).resolve().parents[1]

def acquire(path=None):
    p=Path(path) if path else ROOT/'runs/benchmark_gpu.lock'
    p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f
