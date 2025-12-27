"""
File-based locking for single-instance daemon enforcement.

Prevents multiple instances of the Kobe trading system from running
simultaneously, which could cause duplicate orders or state corruption.

Compatible with both Windows (msvcrt) and Unix (fcntl).
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default lock file location
DEFAULT_LOCK_PATH = Path("state/kobe.lock")

# Maximum age in seconds before a lock is considered stale
STALE_LOCK_AGE_SECONDS = 300  # 5 minutes


class LockError(Exception):
    """Raised when a lock cannot be acquired."""
    pass


class FileLock:
    """
    Cross-platform file-based lock for single-instance enforcement.

    Features:
    - Stale lock detection (auto-release if process died)
    - PID tracking for debugging
    - Context manager support
    - Non-blocking acquire option

    Example:
        lock = FileLock("state/kobe.lock")
        if lock.acquire():
            try:
                # Critical section
                run_trading_loop()
            finally:
                lock.release()

        # Or with context manager:
        with FileLock("state/kobe.lock"):
            run_trading_loop()
    """

    def __init__(
        self,
        lock_path: str | Path = DEFAULT_LOCK_PATH,
        stale_age_seconds: int = STALE_LOCK_AGE_SECONDS,
    ):
        self.lock_path = Path(lock_path)
        self.stale_age_seconds = stale_age_seconds
        self._fd: Optional[int] = None
        self._acquired = False

    def _is_lock_stale(self) -> bool:
        """Check if existing lock file is stale (old and process dead)."""
        if not self.lock_path.exists():
            return False

        try:
            # Check file age
            mtime = self.lock_path.stat().st_mtime
            age = time.time() - mtime
            if age < self.stale_age_seconds:
                return False

            # Check if PID in lock file is still running
            content = self.lock_path.read_text().strip()
            if content.startswith("PID:"):
                pid = int(content.split(":")[1])
                if self._is_process_running(pid):
                    return False

            logger.warning(f"Lock file is stale ({age:.0f}s old), removing")
            return True

        except (OSError, ValueError, IndexError) as e:
            logger.debug(f"Error checking lock staleness: {e}")
            return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        else:
            # Unix: check if process exists
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def _remove_stale_lock(self) -> None:
        """Remove a stale lock file."""
        try:
            self.lock_path.unlink()
            logger.info("Removed stale lock file")
        except OSError as e:
            logger.warning(f"Failed to remove stale lock: {e}")

    def acquire(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: If True, wait for lock. If False, return immediately.
            timeout: Maximum wait time in seconds (only if blocking=True)

        Returns:
            True if lock acquired, False otherwise (only if blocking=False)

        Raises:
            LockError: If blocking=True and lock cannot be acquired within timeout
        """
        # Create parent directory if needed
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for stale lock
        if self._is_lock_stale():
            self._remove_stale_lock()

        start_time = time.time()

        while True:
            try:
                # Try to create lock file exclusively
                self._fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )

                # Write PID to lock file
                os.write(self._fd, f"PID:{os.getpid()}\n".encode())
                os.fsync(self._fd)

                # On Windows, use msvcrt to lock; on Unix, use fcntl
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                self._acquired = True
                logger.info(f"Lock acquired: {self.lock_path}")
                return True

            except (OSError, FileExistsError) as e:
                # Lock file already exists
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except OSError:
                        pass
                    self._fd = None

                if not blocking:
                    return False

                if (time.time() - start_time) >= timeout:
                    raise LockError(f"Could not acquire lock {self.lock_path} within {timeout}s")

                # Wait and retry
                time.sleep(0.5)

    def release(self) -> None:
        """Release the lock."""
        if not self._acquired:
            return

        try:
            if self._fd is not None:
                # Unlock
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    except OSError:
                        pass

                os.close(self._fd)
                self._fd = None

            # Remove lock file
            if self.lock_path.exists():
                self.lock_path.unlink()

            logger.info(f"Lock released: {self.lock_path}")

        except OSError as e:
            logger.warning(f"Error releasing lock: {e}")

        finally:
            self._acquired = False

    def is_locked(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._acquired

    def touch(self) -> None:
        """Update lock file mtime to prevent stale detection."""
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.touch()
            except OSError:
                pass

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise LockError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def __del__(self):
        """Destructor - ensure lock is released."""
        if self._acquired:
            self.release()


@contextmanager
def acquire_lock(
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    blocking: bool = True,
    timeout: float = 10.0,
):
    """
    Context manager for acquiring a file lock.

    Example:
        with acquire_lock("state/kobe.lock"):
            run_trading_loop()
    """
    lock = FileLock(lock_path)
    try:
        lock.acquire(blocking=blocking, timeout=timeout)
        yield lock
    finally:
        lock.release()


def is_another_instance_running(lock_path: str | Path = DEFAULT_LOCK_PATH) -> bool:
    """
    Check if another instance is running without blocking.

    Returns:
        True if another instance holds the lock, False otherwise
    """
    lock = FileLock(lock_path)
    try:
        if lock.acquire(blocking=False):
            lock.release()
            return False
        return True
    except LockError:
        return True
