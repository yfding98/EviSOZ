#!/usr/bin/env python3
"""Launch the read-only multi-dataset EEG waveform viewer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
import webbrowser
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# This host's MNE/numba combination needs JIT disabled before importing MNE.
# The viewer performs I/O and display decimation, not numba-accelerated model
# inference, so this does not change the signal values returned by the API.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eeg_dataset_viewer_mpl")

from eeg_dataset_viewer.datasets import DatasetCatalog, build_dataset_specs  # noqa: E402
from eeg_dataset_viewer.server import make_server  # noqa: E402


DEFAULT_DATA_ROOT = (
    os.environ.get("EEG_DATA_ROOT")
    or (
        "/mnt/hd1/dyf/dataset"
        if Path("/mnt/hd1/dyf/dataset").is_dir()
        else ""
    )
)

_PROC_ROOT = Path("/proc")
_LOCK_PARENT = Path("/tmp")


@dataclass(frozen=True)
class _ViewerProcessIdentity:
    """A listener identity stable enough to guard a single SIGTERM."""

    pid: int
    port: int
    start_time: str
    listener_inodes: frozenset[str]
    script_path: Path


def _listener_inodes(port: int, *, proc_root: Path = _PROC_ROOT) -> frozenset[str]:
    """Return Linux socket inodes listening on ``port`` in this namespace."""

    if not 1 <= int(port) <= 65535:
        return frozenset()
    found_table = False
    inodes: set[str] = set()
    for relative in ("net/tcp", "net/tcp6"):
        table = proc_root / relative
        try:
            lines = table.read_text(encoding="ascii", errors="strict").splitlines()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"无法检查端口 {port} 的监听进程") from exc
        found_table = True
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # TCP_LISTEN
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    if not found_table:
        raise RuntimeError("当前系统不提供 /proc/net/tcp，无法安全识别旧查看器")
    return frozenset(inodes)


def _owners_for_inodes(
    inodes: frozenset[str],
    *,
    proc_root: Path = _PROC_ROOT,
) -> dict[str, frozenset[int]]:
    """Map socket inodes to every visible owning PID without external tools."""

    owners: defaultdict[str, set[int]] = defaultdict(set)
    if not inodes:
        return {}
    targets = {f"socket:[{inode}]": inode for inode in inodes}
    try:
        processes = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("无法遍历 /proc，拒绝终止未知端口进程") from exc
    for process in processes:
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        try:
            descriptors = tuple((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            inode = targets.get(target)
            if inode is not None:
                owners[inode].add(pid)
    return {inode: frozenset(values) for inode, values in owners.items()}


def _process_start_time(process: Path) -> str:
    raw = (process / "stat").read_text(encoding="ascii", errors="strict")
    closing = raw.rfind(")")
    fields = raw[closing + 1 :].split() if closing >= 0 else []
    # The tail starts at field 3 (state); process starttime is field 22.
    if len(fields) <= 19:
        raise RuntimeError("进程 stat 缺少 starttime")
    return fields[19]


def _process_runs_script(process: Path, expected_script: Path) -> bool:
    try:
        executable = (process / "exe").resolve(strict=True)
        expected_executable = Path(sys.executable).resolve(strict=True)
        raw_argv = (process / "cmdline").read_bytes().split(b"\0")
        if raw_argv and raw_argv[-1] == b"":
            raw_argv.pop()  # Remove only /proc's terminator; retain real empty argv.
        argv = [os.fsdecode(value) for value in raw_argv]
        cwd = (process / "cwd").resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return False
    if executable != expected_executable:
        # argv can be arbitrarily shaped by a non-Python program.  Require the
        # exact interpreter of this runner before interpreting Python argv.
        return False
    entry: str | None = None
    # ``exe`` is the Python interpreter, so argv[0] is only a caller-controlled
    # process name.  Linux shebang execution still places the script in
    # argv[1]; never authorize a forged ``exec -a ...`` argv[0].
    arguments = iter(argv[1:])
    safe_interpreter_flags = {
        "-B", "-E", "-I", "-O", "-OO", "-q", "-s", "-S", "-u", "-v", "-x",
    }
    for argument in arguments:
        if argument == "--":
            entry = next(arguments, None)
            break
        if argument.startswith("-") and argument not in safe_interpreter_flags:
            # Fail closed for -cCODE/-mMODULE, clustered flags such as
            # -ucCODE, options carrying a following value, and unknown
            # interpreter modes.  A later .py token may only be data argv.
            return False
        if argument in safe_interpreter_flags:
            continue
        entry = argument
        break
    if not entry or Path(entry).suffix.casefold() != ".py":
        return False
    candidate = Path(entry)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve(strict=True) == expected_script
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _raw_pidfd_syscall(name: str, *arguments: Any) -> int:
    """Call Linux pidfd syscalls when this Python build omits wrappers."""

    import ctypes
    import platform

    # pidfd_send_signal=424 and pidfd_open=434 use the generic Linux syscall
    # allocation on these architectures.  Unknown tables fail closed.
    supported = {"aarch64", "arm64", "x86_64", "amd64"}
    if platform.system() != "Linux" or platform.machine().casefold() not in supported:
        raise RuntimeError("当前平台无法使用 pidfd，拒绝非原子地终止旧查看器")
    number = {"send": 424, "open": 434}[name]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = int(libc.syscall(ctypes.c_long(number), *arguments))
    if result >= 0:
        return result
    error = ctypes.get_errno()
    if error == errno.ESRCH:
        raise ProcessLookupError(error, os.strerror(error))
    if error in {errno.EACCES, errno.EPERM}:
        raise PermissionError(error, os.strerror(error))
    raise OSError(error, os.strerror(error))


def _open_pidfd(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return int(native(pid, 0))
    import ctypes

    return _raw_pidfd_syscall("open", ctypes.c_int(pid), ctypes.c_uint(0))


def _send_pidfd_signal(pidfd: int, signum: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(pidfd, signum, None, 0)
        return
    import ctypes

    _raw_pidfd_syscall(
        "send",
        ctypes.c_int(pidfd),
        ctypes.c_int(signum),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )


def _verified_viewer_owner(
    port: int,
    *,
    expected_script: Path | None = None,
    proc_root: Path = _PROC_ROOT,
) -> _ViewerProcessIdentity:
    """Fail closed unless one same-user listener is this exact runner script."""

    script_path = (expected_script or Path(__file__)).resolve(strict=True)
    inodes = _listener_inodes(port, proc_root=proc_root)
    if not inodes:
        raise RuntimeError(
            f"端口 {port} 已被占用，但没有可安全验证的监听 socket；拒绝终止进程"
        )
    owners = _owners_for_inodes(inodes, proc_root=proc_root)
    if any(len(owners.get(inode, ())) != 1 for inode in inodes):
        raise RuntimeError(f"端口 {port} 的监听进程不唯一或不可见；拒绝终止")
    pids = {pid for inode in inodes for pid in owners[inode]}
    if len(pids) != 1:
        raise RuntimeError(f"端口 {port} 由多个进程共享；拒绝终止")
    pid = next(iter(pids))
    if pid <= 1 or pid == os.getpid():
        raise RuntimeError(f"端口 {port} 的监听者不是可替换的旧查看器")
    process = proc_root / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            raise RuntimeError(f"端口 {port} 由其他用户进程占用；拒绝终止")
        start_time = _process_start_time(process)
    except FileNotFoundError:
        raise RuntimeError(f"端口 {port} 的旧监听进程已变化，请重试") from None
    except OSError as exc:
        raise RuntimeError(f"无法验证端口 {port} 的监听进程；拒绝终止") from exc
    if not _process_runs_script(process, script_path):
        raise RuntimeError(
            f"端口 {port} 被非本 EEG 查看器进程占用；为避免误杀，启动已中止"
        )
    return _ViewerProcessIdentity(
        pid=pid,
        port=port,
        start_time=start_time,
        listener_inodes=inodes,
        script_path=script_path,
    )


def _identity_still_owns_listener(
    identity: _ViewerProcessIdentity,
    *,
    proc_root: Path = _PROC_ROOT,
) -> bool:
    process = proc_root / str(identity.pid)
    try:
        if process.stat().st_uid != os.getuid():
            return False
        if _process_start_time(process) != identity.start_time:
            return False
        if not _process_runs_script(process, identity.script_path):
            return False
        current = _listener_inodes(identity.port, proc_root=proc_root)
    except (FileNotFoundError, OSError, RuntimeError):
        return False
    active = identity.listener_inodes & current
    if not active:
        return False
    owners = _owners_for_inodes(frozenset(active), proc_root=proc_root)
    return all(owners.get(inode) == frozenset({identity.pid}) for inode in active)


def _terminate_verified_viewer(
    identity: _ViewerProcessIdentity,
    *,
    timeout: float,
    proc_root: Path = _PROC_ROOT,
    kill_fn: Callable[[int, int], Any] | None = None,
    sleep_fn: Callable[[float], Any] = time.sleep,
    pidfd_open_fn: Callable[[int], int] = _open_pidfd,
    pidfd_send_fn: Callable[[int, int], Any] = _send_pidfd_signal,
    close_fn: Callable[[int], Any] = os.close,
) -> bool:
    """Send one PID-pinned SIGTERM and wait for the listener to release."""

    pidfd: int | None = None
    try:
        if kill_fn is None:
            # Opening first pins this process object.  The subsequent
            # starttime/inode recheck then proves the handle is the verified
            # listener, eliminating the PID-reuse window of os.kill(pid, ...).
            pidfd = pidfd_open_fn(identity.pid)
        if not _identity_still_owns_listener(identity, proc_root=proc_root):
            return False
        if kill_fn is None:
            pidfd_send_fn(pidfd, signal.SIGTERM)
        else:
            # Test injection only; production always takes the pidfd path.
            kill_fn(identity.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(
            f"没有权限终止端口 {identity.port} 上的旧 EEG 查看器"
        ) from exc
    except OSError as exc:
        raise RuntimeError("当前内核无法安全使用 pidfd 终止旧查看器") from exc
    finally:
        if pidfd is not None:
            close_fn(pidfd)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _identity_still_owns_listener(identity, proc_root=proc_root):
            return True
        sleep_fn(0.05)
    raise RuntimeError(
        f"旧 EEG 查看器未在 {timeout:g} 秒内释放端口 {identity.port}；"
        "未强制 SIGKILL，请手动检查该进程"
    )


@contextmanager
def _startup_port_lock(
    port: int,
    *,
    expected_script: Path | None = None,
    lock_parent: Path = _LOCK_PARENT,
    timeout: float = 5.0,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> Iterator[None]:
    """Serialize bind/inspect/terminate/rebind for this script and port."""

    import fcntl

    script_path = (expected_script or Path(__file__)).resolve(strict=True)
    directory = lock_parent / f"eeg-dataset-viewer-{os.getuid()}"
    directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeError(f"启动锁目录不安全: {directory}")
    digest = hashlib.sha256(os.fsencode(str(script_path))).hexdigest()[:16]
    lock_path = directory / f"{digest}-{port}.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        lock_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_uid != os.getuid():
            raise RuntimeError(f"启动锁文件不安全: {lock_path}")
        os.fchmod(descriptor, 0o600)
        deadline = clock_fn() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if clock_fn() >= deadline:
                    raise RuntimeError(
                        f"等待端口 {port} 的启动锁超过 {timeout:g} 秒"
                    ) from None
                sleep_fn(0.05)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _make_server_replacing_existing_viewer(
    catalog: DatasetCatalog,
    *,
    host: str,
    port: int,
    access_token: str | None,
    verbose: bool,
    replace_existing: bool,
    replace_timeout: float,
    server_factory: Callable[..., Any] = make_server,
    proc_root: Path = _PROC_ROOT,
    lock_parent: Path = _LOCK_PARENT,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], Any] = time.sleep,
    on_bound: Callable[[Any], Any] | None = None,
) -> Any:
    def bind() -> Any:
        return server_factory(
            catalog,
            host=host,
            port=port,
            access_token=access_token,
            verbose=verbose,
        )

    def prepare_bound(server: Any) -> Any:
        if on_bound is None:
            return server
        try:
            on_bound(server)
        except Exception:
            try:
                server.server_close()
            finally:
                raise
        return server

    if port == 0 or not replace_existing:
        return prepare_bound(bind())
    with _startup_port_lock(
        port,
        lock_parent=lock_parent,
        timeout=replace_timeout,
        clock_fn=clock_fn,
        sleep_fn=sleep_fn,
    ):
        try:
            return prepare_bound(bind())
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
        identity = _verified_viewer_owner(port, proc_root=proc_root)
        terminated = _terminate_verified_viewer(
            identity,
            timeout=replace_timeout,
            proc_root=proc_root,
        )
        if terminated:
            print(f"Port {port}: stopped previous EEG dataset viewer (pid {identity.pid})")
        else:
            print(
                f"Port {port}: previous listener changed before SIGTERM; "
                "no signal was sent"
            )
        # A graceful server_close can trail the listener-inode disappearance
        # by a few scheduler ticks.  Retry binding for the bounded replacement
        # window, but never inspect or signal a second owner: if another
        # process wins the race, this loop only waits and then fails closed.
        deadline = clock_fn() + replace_timeout
        while True:
            try:
                return prepare_bound(bind())
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                if clock_fn() >= deadline:
                    raise RuntimeError(
                        f"旧查看器退出后端口 {port} 仍被占用；"
                        "拒绝继续终止新的监听进程"
                    ) from None
                sleep_fn(0.05)


def _install_graceful_sigterm(server: Any) -> Callable[[], None]:
    """Install early shutdown handling and return an idempotent restorer."""

    previous = signal.getsignal(signal.SIGTERM)
    requested = threading.Event()
    restored = threading.Event()

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        if requested.is_set():
            return
        requested.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_sigterm)

    def restore() -> None:
        if restored.is_set():
            return
        restored.set()
        if signal.getsignal(signal.SIGTERM) is handle_sigterm:
            signal.signal(signal.SIGTERM, previous)

    return restore


@contextmanager
def _graceful_sigterm(server: Any) -> Iterator[None]:
    """Compatibility context used by focused tests and callers."""

    restore = _install_graceful_sigterm(server)
    try:
        yield
    finally:
        restore()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只读扫描并可视化 TUSZ/TUEV/TUAR/TUAB/TUEG/CHB-MIT/"
            "Siena/VEPiSet、私有或通用 EEG 数据。"
        )
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="公开数据集父目录；自动识别常见下载目录。",
    )
    parser.add_argument(
        "--no-auto-discover",
        action="store_true",
        help="不从 --data-root 自动发现公开数据集，只使用显式 root。",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help=(
            "添加数据集根目录，可重复。KIND 为 tusz/tuev/tuar/tuab/tueg/"
            "chbmit/siena/vepiset/private/generic。"
        ),
    )
    for flag, kind, label in (
        ("--tusz-root", "tusz", "TUSZ"),
        ("--tuev-root", "tuev", "TUEV"),
        ("--tuar-root", "tuar", "TUAR"),
        ("--tuab-root", "tuab", "TUAB"),
        ("--tueg-root", "tueg", "TUEG"),
        ("--chbmit-root", "chbmit", "CHB-MIT"),
        ("--siena-root", "siena", "Siena Scalp EEG"),
        ("--vepiset-root", "vepiset", "VEPiSet"),
    ):
        parser.add_argument(
            flag,
            action="append",
            default=[],
            metavar="PATH",
            help=f"添加 {label} 根目录，可重复。",
        )
    parser.add_argument(
        "--private-root",
        action="append",
        default=[],
        metavar="PATH",
        help="添加私有 EEG 根目录，可重复；不会由 --data-root 自动推断。",
    )
    parser.add_argument(
        "--annotation-manifest",
        action="append",
        default=[],
        metavar="DATASET=FILE",
        help=(
            "为一个数据集添加只读 CSV/TSV/JSON/JSONL 标注清单，可重复。"
            "DATASET 使用页面中的数据集 key。"
        ),
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="调试用：每个数据集最多索引多少个 EEG 文件。",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=4,
        help="最多缓存多少个 lazy Raw header（默认 4，不预载完整波形）。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--replace-existing-viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "端口被本脚本启动的旧 EEG 查看器占用时，先安全发送 SIGTERM "
            "再启动（默认开启；用 --no-replace-existing-viewer 关闭）。"
        ),
    )
    parser.add_argument(
        "--replace-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="等待旧 EEG 查看器释放端口的秒数（默认 5，绝不自动 SIGKILL）。",
    )
    parser.add_argument(
        "--access-token",
        default=os.environ.get("EEG_VIEWER_TOKEN"),
        help=(
            "非 loopback 监听时必填；也可通过 EEG_VIEWER_TOKEN 提供。"
            "Basic Auth 用户名固定为 viewer。"
        ),
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="服务启动后用系统默认浏览器打开页面。",
    )
    parser.add_argument(
        "--print-catalog",
        action="store_true",
        help="打印扫描后的目录 JSON 并退出，不启动 HTTP 服务。",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _explicit_roots(args: argparse.Namespace) -> list[str]:
    roots = list(args.dataset_root)
    for kind in (
        "tusz", "tuev", "tuar", "tuab", "tueg", "chbmit", "siena", "vepiset",
    ):
        for path in getattr(args, f"{kind}_root"):
            roots.append(f"{kind}={path}")
    roots.extend(f"private={path}" for path in args.private_root)
    return roots


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_files_per_dataset is not None and args.max_files_per_dataset < 1:
        raise SystemExit("--max-files-per-dataset 必须大于 0")
    if args.cache_size < 1 or args.cache_size > 32:
        raise SystemExit("--cache-size 必须在 1 到 32 之间")
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port 必须在 0 到 65535 之间")
    if (
        not math.isfinite(args.replace_timeout)
        or not 0.1 <= args.replace_timeout <= 30.0
    ):
        raise SystemExit("--replace-timeout 必须在 0.1 到 30 秒之间")

    specs = build_dataset_specs(
        data_root=None if args.no_auto_discover else (args.data_root or None),
        dataset_roots=_explicit_roots(args),
        annotation_manifests=args.annotation_manifest,
    )
    if not specs:
        raise SystemExit(
            "没有找到数据集。请使用 --data-root，或传入例如 "
            "--tusz-root /path/to/TUSZ/edf、--siena-root /path/to/Siena，"
            "或 --private-root /path/to/private。"
        )

    catalog = DatasetCatalog(
        specs,
        max_files_per_dataset=args.max_files_per_dataset,
        cache_size=args.cache_size,
    )
    server = None
    restore_sigterm: Callable[[], None] | None = None

    def install_sigterm(bound_server: Any) -> None:
        nonlocal restore_sigterm
        restore_sigterm = _install_graceful_sigterm(bound_server)

    try:
        metadata = catalog.catalog_metadata()
        if int(metadata.get("record_count", 0)) <= 0:
            raise SystemExit("已识别数据集目录，但其中没有找到受支持的 EEG 文件。")
        if args.print_catalog:
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return 0

        try:
            server = _make_server_replacing_existing_viewer(
                catalog,
                host=args.host,
                port=args.port,
                access_token=args.access_token,
                verbose=args.verbose,
                replace_existing=args.replace_existing_viewer,
                replace_timeout=args.replace_timeout,
                on_bound=install_sigterm,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
        bound_host, bound_port = server.server_address[:2]
        browser_host = "127.0.0.1" if str(bound_host) in {"0.0.0.0", "::"} else bound_host
        url = f"http://{browser_host}:{bound_port}/"
        print(f"EEG dataset viewer: {url}")
        print(f"Indexed {metadata['record_count']} recording(s) in {len(metadata['datasets'])} dataset(s)")
        print("Read-only mode: source EEG and annotation files will not be modified")
        if args.access_token:
            print("HTTP Basic Auth username: viewer")
        if args.open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            print("\nStopping EEG dataset viewer")
    finally:
        try:
            if restore_sigterm is not None:
                restore_sigterm()
        finally:
            try:
                if server is not None:
                    server.server_close()
            finally:
                catalog.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
