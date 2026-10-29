"""Windows のプロセス寿命管理。

- 子プロセス(llama-server)を Job Object に入れ、親の Python がどんな形で
  終了しても(コンソールの×・強制終了・クラッシュ)OS に道連れで終了させる
- 以前の実行で取り残された llama-server を起動前に終了し、VRAM を空ける
- コンソールの QuickEdit を切り、クリックで出力が止まって処理が固まるのを防ぐ
"""
import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_TH32CS_SNAPPROCESS = 0x2
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_ERROR_INVALID_PARAMETER = 87
_WAIT_OBJECT_0 = 0
_STD_INPUT_HANDLE = -10
_ENABLE_QUICK_EDIT_MODE = 0x0040
_ENABLE_EXTENDED_FLAGS = 0x0080

# ハンドルを閉じるとジョブ内のプロセスが終了するため、親プロセスの寿命まで保持する。
_job_handle = None


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _ProcessEntry32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260)]


if _IS_WINDOWS:
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32)]
    _kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32)]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetStdHandle.restype = wintypes.HANDLE
    _kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    _kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def _kill_on_close_job():
    global _job_handle
    if _job_handle is None:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
                job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(job)
            raise ctypes.WinError(error)
        _job_handle = job
    return _job_handle


def bind_child_lifetime(process: subprocess.Popen):
    """親プロセスが終了したら process も OS に終了させる。"""
    if not _IS_WINDOWS:
        return
    try:
        if not _kernel32.AssignProcessToJobObject(_kill_on_close_job(), int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    except OSError as exc:
        print(f"[proc] 子プロセスの自動終了設定に失敗しました: {exc}")


def _creation_time(handle) -> int | None:
    times = [wintypes.FILETIME() for _ in range(4)]
    if not _kernel32.GetProcessTimes(handle, *map(ctypes.byref, times)):
        return None
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def _process_path(handle) -> str | None:
    size = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
        return None
    return buffer.value


def _snapshot_processes() -> list[tuple[int, int, str]]:
    snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == _INVALID_HANDLE_VALUE:
        return []
    processes = []
    try:
        entry = _ProcessEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            processes.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            ok = _kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        _kernel32.CloseHandle(snapshot)
    return processes


def _parent_is_alive(parent_pid: int, child_created: int | None) -> bool:
    handle = _kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, parent_pid)
    if not handle:
        # 存在しない PID だけを「親なし」とみなす。権限不足等は生存扱いにして触らない。
        return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
    try:
        # 終了済みでも、他プロセスがハンドルを持っている間は開けてしまう。
        if _kernel32.WaitForSingleObject(handle, 0) == _WAIT_OBJECT_0:
            return False
        parent_created = _creation_time(handle)
    finally:
        _kernel32.CloseHandle(handle)
    # PID が再利用されていれば「親」は子より後に起動している。
    return not (parent_created and child_created and parent_created > child_created)


def kill_orphaned_processes(exe_dir: Path, exe_name: str) -> int:
    """exe_dir 配下の exe_name のうち、親プロセスが既に存在しないものを終了する。"""
    if not _IS_WINDOWS:
        return 0
    root = str(exe_dir.resolve()).lower().rstrip("\\") + "\\"
    killed = 0
    for pid, parent_pid, name in _snapshot_processes():
        if name.lower() != exe_name.lower():
            continue
        handle = _kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _SYNCHRONIZE, False, pid)
        if not handle:
            continue
        try:
            path = _process_path(handle)
            if not path or not path.lower().startswith(root):
                continue
            if _parent_is_alive(parent_pid, _creation_time(handle)):
                continue
            if _kernel32.TerminateProcess(handle, 1):
                _kernel32.WaitForSingleObject(handle, 5000)
                killed += 1
                print(f"[proc] 前回の実行で残っていた {exe_name} (PID {pid}) を終了しました")
        finally:
            _kernel32.CloseHandle(handle)
    return killed


def disable_console_quick_edit():
    """コンソールのクリックで出力がブロックされ、処理全体が止まるのを防ぐ。

    元の設定に戻す関数を返す(終了後もコンソールで文字を選択できるように)。
    """
    if not _IS_WINDOWS:
        return lambda: None
    handle = _kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    mode = wintypes.DWORD()
    if not handle or not _kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return lambda: None  # コンソールに接続されていない(リダイレクト等)
    original = mode.value
    new_mode = (original & ~_ENABLE_QUICK_EDIT_MODE) | _ENABLE_EXTENDED_FLAGS
    if new_mode != original:
        _kernel32.SetConsoleMode(handle, new_mode)
    return lambda: _kernel32.SetConsoleMode(handle, original)
