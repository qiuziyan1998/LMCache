# SPDX-License-Identifier: Apache-2.0
"""Final-access ownership for ordinary native prepared-source consumers."""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable


@dataclass
class _NativeSource:
    source: Any
    hold: Any
    streams: dict[tuple[int, int], Any] = field(default_factory=dict)
    events: tuple[Any, ...] = ()
    quarantined: bool = False
    fences_ready: bool = False


class NativeSourceLifetime:
    """Retain once per source, fencing each participating stream at removal.

    Native steps only record the participating stream in CPU metadata. Device
    events are created at source retirement, after its final submitted access.
    Graph replay owns its separate existing latest-event child leases.
    """

    def __init__(
        self, event_factory: Callable[[], Any], fatal: Callable[[str], None]
    ) -> None:
        self._event_factory = event_factory
        self._fatal = fatal
        self._lock = RLock()
        self._active: dict[int, _NativeSource] = {}
        self._retiring: dict[int, _NativeSource] = {}

    def note(self, source: Any, stream: Any) -> None:
        """Acquire before native submission and remember its actual stream."""
        if source.owner_lease is None:
            return
        key = id(source.binding_token)
        with self._lock:
            if key in self._retiring:
                raise RuntimeError("Native source was submitted after retirement")
            record = self._active.get(key)
            if record is None:
                hold = source.owner_lease.retain()
                hold.mark_imported()
                record = _NativeSource(source, hold)
                self._active[key] = record
            stream_id = getattr(stream, "npu_stream", None)
            stream_key = (
                int(getattr(stream, "device_index", -1)),
                id(stream) if stream_id is None else int(stream_id),
            )
            record.streams[stream_key] = stream

    def retire_source(self, source: Any) -> None:
        """Record final stream fences before dropping a request's source hold."""
        key = id(source.binding_token)
        with self._lock:
            record = self._active.pop(key, None)
            if record is None:
                return
            self._retiring[key] = record
        events = []
        try:
            for stream in record.streams.values():
                event = self._event_factory()
                events.append(event)
                event.record(stream)
            record.events = tuple(events)
            record.fences_ready = True
        except BaseException:
            record.events = tuple(events)
            record.quarantined = True
            self._fatal("Native source final-access event could not be recorded")
            raise
        self.drain()

    def drain(self) -> None:
        """Query pending final accesses without a host or device wait."""
        with self._lock:
            pending = tuple(self._retiring.items())
        for key, record in pending:
            if record.quarantined or not record.fences_ready:
                continue
            try:
                ready = all(event.query() for event in record.events)
            except BaseException:
                record.quarantined = True
                self._fatal("Native source final-access completion is unknown")
                raise
            if ready:
                with self._lock:
                    if self._retiring.get(key) is not record:
                        continue
                    record.hold.release("fenced")
                    del self._retiring[key]

    @property
    def retirements_pending(self) -> bool:
        """Check pending work without events, device queries or cache traversal."""
        return bool(self._retiring)

    def sources_drained(self) -> bool:
        """Check local completion before any mapping may be unregistered."""
        self.drain()
        with self._lock:
            return not self._active and not self._retiring

    def close(self) -> None:
        """Fence known shutdown work, retaining every uncertain native owner."""
        with self._lock:
            active = tuple(record.source for record in self._active.values())
        for source in active:
            self.retire_source(source)
        with self._lock:
            pending = tuple(self._retiring.values())
        for record in pending:
            if record.quarantined:
                raise RuntimeError("Native source completion is unknown")
            try:
                for event in record.events:
                    event.synchronize()
            except BaseException:
                record.quarantined = True
                self._fatal("Native source shutdown fence failed")
                raise
        self.drain()
