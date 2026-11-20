# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, Sequence
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryObj


logger = init_logger(__name__)


class PinMonitor(PeriodicThread):
    """
    Global monitor (singleton per process, shared across all cache engines)
    for pinned TensorMemoryObj instances to handle timeout detection.
    This class runs a background thread that periodically checks for pinned objects
    that have exceeded their timeout duration.
    """

    _instance = None
    _instance_lock = threading.Lock()  # Class-level lock for singleton pattern

    def __init__(self, config: "LMCacheEngineConfig"):
        # Initialize PeriodicThread base class
        super().__init__(
            name="PinMonitor-thread",
            interval=config.pin_check_interval_sec,
            level=ThreadLevel.CRITICAL,
            init_wait=0.0,
        )

        # obj_id is the virtual memory address given by Python's id() function
        self._pinned_objects: dict[
            int, tuple["MemoryObj", float]
        ] = {}  # {obj_id: (memory_obj, register_time)}
        self._pin_generations: dict[int, int] = {}
        self._pin_seq: int = 0
        self._objects_lock = threading.Lock()
        self._check_interval = config.pin_check_interval_sec
        self._pin_timeout_sec = config.pin_timeout_sec

        # Register with the global registry
        PeriodicThreadRegistry.get_instance().register(self)

        # Auto-start the monitor on first instance creation
        self.start_monitoring()

    @staticmethod
    def GetOrCreate(config: Optional["LMCacheEngineConfig"] = None) -> "PinMonitor":
        """Get or create the singleton instance.

        Args:
            config: Required for first-time initialization.
                Optional for subsequent calls.

        Raises:
            ValueError: If config is None when creating the instance
                for the first time.
        """
        if PinMonitor._instance is None:
            with PinMonitor._instance_lock:
                if PinMonitor._instance is None:
                    assert config is not None, "config is required"
                    PinMonitor._instance = PinMonitor(config)
        return PinMonitor._instance

    def on_pin(self, memory_obj: "MemoryObj"):
        """Register a pinned memory object for timeout monitoring.

        Note: The same memory_obj can be pinned multiple times, so this
        function may be called multiple times with the same object.
        Each call updates the register time, effectively resetting the
        timeout countdown.
        """
        obj_id = id(memory_obj)
        with self._objects_lock:
            current_time = time.time()
            self._pin_seq += 1
            self._pinned_objects[obj_id] = (memory_obj, current_time)
            self._pin_generations[obj_id] = self._pin_seq
            logger.debug(
                "Registered pinned object %s for timeout monitoring at time %.2f",
                obj_id,
                current_time,
            )

    def on_pin_many(self, memory_objs: Sequence["MemoryObj"]):
        """Register a batch with one timestamp and one monitor lock."""
        with self._objects_lock:
            current_time = time.time()
            seq = self._pin_seq
            for obj in memory_objs:
                seq += 1
                oid = id(obj)
                self._pinned_objects[oid] = (obj, current_time)
                self._pin_generations[oid] = seq
            self._pin_seq = seq
            if logger.isEnabledFor(10):
                logger.debug(
                    "Registered %d pinned objects for timeout monitoring",
                    len(memory_objs),
                )

    def on_unpin(self, memory_obj: "MemoryObj"):
        """Unregister a memory object from timeout monitoring."""
        obj_id = id(memory_obj)
        with self._objects_lock:
            if obj_id in self._pinned_objects:
                del self._pinned_objects[obj_id]
                self._pin_generations.pop(obj_id, None)
                if logger.isEnabledFor(10):
                    logger.debug(
                        "Unregistered pinned object %s from timeout monitoring",
                        obj_id,
                    )

    def on_unpin_many(self, memory_objs: Sequence["MemoryObj"]) -> None:
        """Unregister a batch of pinned objects under one monitor lock."""
        if not memory_objs:
            return
        with self._objects_lock:
            pinned = self._pinned_objects
            gens = self._pin_generations
            removed = 0
            for obj in memory_objs:
                oid = id(obj)
                if (
                    getattr(getattr(obj, "meta", None), "pin_count", 0) <= 0
                    and pinned.pop(oid, None) is not None
                ):
                    gens.pop(oid, None)
                    removed += 1
            if removed and logger.isEnabledFor(10):
                logger.debug(
                    "Unregistered %d pinned objects from timeout monitoring",
                    removed,
                )

    def _check_timeouts(self) -> tuple[int, int, int]:
        """Check all registered pinned objects for timeout.

        Returns:
            tuple: (pinned_count, timeout_count, force_unpin_success_count)
        """
        current_time = time.time()
        timeout_objects: list[tuple["MemoryObj", float, Optional[int]]] = []

        with self._objects_lock:
            pinned_count = len(self._pinned_objects)
            for obj_id, (memory_obj, register_time) in list(
                self._pinned_objects.items()
            ):
                # Check if object is still pinned and has exceeded timeout
                if memory_obj.meta.pin_count > 0:
                    elapsed_time = current_time - register_time
                    if elapsed_time > self._pin_timeout_sec:
                        timeout_objects.append(
                            (
                                memory_obj,
                                elapsed_time,
                                self._pin_generations.get(obj_id),
                            )
                        )

        # Force unpin timeout objects outside the lock to avoid deadlocks
        force_unpin_success_count = 0
        for memory_obj, elapsed_time, expected_gen in timeout_objects:
            try:
                if self._force_unpin_timeout_object(
                    memory_obj, elapsed_time, expected_gen
                ):
                    force_unpin_success_count += 1
            except Exception as e:
                logger.error(
                    "Error forcing unpin for timeout object %s: %s", id(memory_obj), e
                )
        if force_unpin_success_count > 0:
            logger.warning(
                "Force unpinned %d timeout objects in %d pinned_objects "
                "within %d seconds",
                force_unpin_success_count,
                pinned_count,
                self._pin_timeout_sec,
            )
        else:
            logger.debug(
                "PinMonitor check: pinned_objects=%d, timeout_objects=%d, "
                "force_unpin_success=%d",
                pinned_count,
                len(timeout_objects),
                force_unpin_success_count,
            )

        return pinned_count, len(timeout_objects), force_unpin_success_count

    def _force_unpin_timeout_object(
        self,
        memory_obj: "MemoryObj",
        elapsed_time: float,
        expected_gen: Optional[int] = None,
    ) -> bool:
        """Force unpin a timeout object if its pin generation has not refreshed."""
        # First Party
        from lmcache.v1.memory_management import (
            LayerPageMemoryObj,
            TensorMemoryObj,
        )

        obj_id = id(memory_obj)
        if (
            isinstance(memory_obj, TensorMemoryObj)
            and type(memory_obj) in (TensorMemoryObj, LayerPageMemoryObj)
            and "unpin" not in memory_obj.__dict__
        ):
            should_free = False
            with memory_obj.lock:
                with self._objects_lock:
                    if expected_gen is not None:
                        cur_entry = self._pinned_objects.get(obj_id)
                        if (
                            cur_entry is None
                            or self._pin_generations.get(obj_id) != expected_gen
                        ):
                            return False
                    current_pin_count = memory_obj.meta.pin_count
                    if current_pin_count <= 0:
                        self._pinned_objects.pop(obj_id, None)
                        self._pin_generations.pop(obj_id, None)
                        return False
                    self._pinned_objects.pop(obj_id, None)
                    self._pin_generations.pop(obj_id, None)
                logger.warning(
                    "Pin timeout detected for MemoryObj %s. "
                    "Pin count: %s, Elapsed time: %.2fs. Forcing unpin to 0.",
                    memory_obj.meta.address,
                    current_pin_count,
                    elapsed_time,
                )
                memory_obj.meta.pin_count = 0
                if (
                    memory_obj.meta.ref_count <= 0
                    and memory_obj.valid
                    and not getattr(memory_obj, "_free_claimed", False)
                ):
                    memory_obj._free_claimed = True
                    should_free = True
            stats = LMCStatsMonitor.GetOrCreate()
            stats.update_pinned_memory_objs_count(-1)
            stats.update_forced_unpin_count(1)
            if should_free:
                alloc = memory_obj.parent_allocator
                if alloc is not None:
                    alloc.free(memory_obj)
            return True

        obj_lock = getattr(memory_obj, "lock", None) or nullcontext()
        with obj_lock:
            with self._objects_lock:
                if (
                    expected_gen is not None
                    and self._pin_generations.get(obj_id) != expected_gen
                ):
                    return False
            current_pin_count = memory_obj.meta.pin_count
            if current_pin_count <= 0:
                return False

            logger.warning(
                "Pin timeout detected for MemoryObj %s. "
                "Pin count: %s, Elapsed time: %.2fs. Forcing unpin to 0.",
                memory_obj.meta.address,
                current_pin_count,
                elapsed_time,
            )

        LMCStatsMonitor.GetOrCreate().update_forced_unpin_count(1)
        while memory_obj.meta.pin_count > 0:
            if expected_gen is not None:
                with self._objects_lock:
                    if self._pin_generations.get(obj_id) != expected_gen:
                        break
            memory_obj.unpin()
        return True

    def _execute(self) -> ThreadRunSummary:
        """
        Execute one pin monitor check cycle.

        This method is called by the PeriodicThread base class.

        Returns:
            ThreadRunSummary: Summary of the check cycle
        """
        pinned_count, timeout_count, force_unpin_count = self._check_timeouts()

        return ThreadRunSummary(
            success=True,
            message=f"Checked {pinned_count} objects, {timeout_count} timeouts, "
            f"{force_unpin_count} force unpinned",
            extra_info={
                "pinned_count": str(pinned_count),
                "timeout_count": str(timeout_count),
                "force_unpin_count": str(force_unpin_count),
            },
        )

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self.is_running:
            return

        # Use base class start method
        self.start()
        logger.info("PinMonitor started")

        # Setup metrics callback
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.pin_monitor_pinned_objects_count.set_function(
                lambda: len(self._pinned_objects)
            )

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        if not self.is_running:
            return

        # Unregister from the global registry
        PeriodicThreadRegistry.get_instance().unregister(self.name)

        # Use base class stop method
        self.stop()
        logger.info("PinMonitor stopped")

    def get_monitored_count(self) -> int:
        """Get the number of currently monitored pinned objects."""
        with self._objects_lock:
            return len(self._pinned_objects)

    @staticmethod
    def DestroyInstance():
        """Destroy the singleton instance and stop monitoring.
        This is mainly used for testing to ensure clean state between tests.
        """
        with PinMonitor._instance_lock:
            if PinMonitor._instance is not None:
                PinMonitor._instance.stop_monitoring()
                PinMonitor._instance = None
