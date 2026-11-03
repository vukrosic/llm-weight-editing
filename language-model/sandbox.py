"""Run the editor's Python where it can only see its work folder. Numpy and scipy only: no torch, no network, no other files.

macOS: sandbox-exec. Linux: bubblewrap. A run gets a clean environment, CPU seconds, a file-size cap and an output cap.
Nothing the code prints or writes is trusted by the host: harness.py checks it before use.
"""
import json, os, platform, resource, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("KERNEL_PY", os.path.join(HERE, ".kernel", "bin", "python"))
MAX_OUT, FILE_MB = 20_000, 512

PRELUDE = r'''
import json, os, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
ARCH = json.load(open("arch.json")); SHAPES = {k: tuple(v) for k, v in ARCH["shapes"].items()}; NAMES = list(SHAPES)
def load(name):
    """The original tensor (read only). Use load_edit for the current edit."""
    return np.load("weights/" + name + ".npy")
def load_edit(name):
    p = "edit/" + name + ".npy"
    return np.load(p) if os.path.exists(p) else load(name)
def save_edit(name, a):
    assert name in SHAPES, f"unknown tensor {name}"
    a = np.asarray(a, np.float32)
    assert a.shape == SHAPES[name], (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), name
    os.makedirs("edit", exist_ok=True); np.save("edit/" + name + ".npy", a)
def drop_edit(name):
    p = "edit/" + name + ".npy"
    if os.path.exists(p): os.remove(p)
'''


def profile(work):
    """Deny everything, then allow the interpreter, system libraries and the work folder. No network, no new processes."""
    base = os.path.realpath(sys_prefix())
    prefix = os.path.realpath(os.path.dirname(os.path.dirname(PY)))
    return f'''(version 1)
(deny default)
(allow process-exec (subpath "{base}") (subpath "{prefix}"))
(allow signal (target self))
(allow sysctl-read)
(allow file-read-metadata)
(allow file-map-executable)
(allow file-read* (literal "/") (subpath "/usr/lib") (subpath "/usr/share") (subpath "/System") (subpath "/private/var/db/dyld")
    (subpath "/dev") (subpath "{base}") (subpath "{prefix}") (subpath "{work}"))
(allow file-write* (subpath "{work}") (literal "/dev/null"))
'''


def sys_prefix():
    out = subprocess.run([PY, "-I", "-c", "import sys; print(sys.base_prefix)"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def command(work):
    if platform.system() == "Darwin":
        return ["sandbox-exec", "-p", profile(work), PY, "-I", "-"]
    if not shutil.which("bwrap"): raise RuntimeError("Linux needs bubblewrap (bwrap) for the editor's sandbox")
    prefix, base = os.path.realpath(os.path.dirname(os.path.dirname(PY))), os.path.realpath(sys_prefix())
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin", "--proc", "/proc",
            "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", prefix, prefix, "--ro-bind", base, base, "--bind", work, work,
            "--chdir", work, PY, "-I", "-"]


def run(code, work, cpu_seconds=30):
    """Returns (printed text, CPU seconds used). The code runs after PRELUDE in the work folder."""
    work = os.path.realpath(work)

    def limits():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_MB << 20, FILE_MB << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    env = {"HOME": work, "PATH": "/usr/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    p = subprocess.Popen(command(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=work,
                         env=env, preexec_fn=limits)
    try:
        out, _ = p.communicate(PRELUDE + "\n" + code, timeout=cpu_seconds * 3 + 5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); out, _ = p.communicate(); out += "\n[stopped at the time limit]"
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    used = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    if p.returncode and p.returncode < 0: out += f"\n[stopped by signal {-p.returncode}: the limit is {cpu_seconds} CPU seconds per run]"
    if len(out) > MAX_OUT: out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    return out or "(no output)", round(used, 3)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        json.dump({"shapes": {"a": [2, 2]}}, open(os.path.join(d, "arch.json"), "w"))
        import numpy as np
        os.makedirs(os.path.join(d, "weights")); np.save(os.path.join(d, "weights", "a.npy"), np.eye(2, dtype=np.float32))
        print(run("print(load('a') @ load('a')); save_edit('a', load('a') * 2); print(os.listdir('edit'))", d))
