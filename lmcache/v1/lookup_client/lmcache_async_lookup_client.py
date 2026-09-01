# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import asyncio
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupMsg,
    LookupCleanupResponseMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)

logger = init_logger(__name__)


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup server on all workers
        - if lookup_server_worker_ids is [0], start lookup server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start lookup server on
          worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        # lookup_id -> first lookup time
        # this helps us support timeout semantics
        self.first_lookup_time: dict[str, float] = {}
        self.config = config

        self.ctx = get_zmq_context(use_asyncio=False)
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        engine_id = metadata.engine_id
        assert engine_id is not None, "engine_id is required for RPC communication"
        self.world_size = metadata.world_size
        self.lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        self.push_sockets = []
        if len(self.lookup_server_worker_ids) > 0:
            ranks = self.lookup_server_worker_ids
            self.world_size = len(self.lookup_server_worker_ids)
        else:
            ranks = [i for i in range(self.world_size)]

        for rank in ranks:
            worker_socket_path = get_zmq_rpc_path_lmcache(
                engine_id, "lookup_worker", rpc_port, rank
            )
            logger.info(
                "lmcache lookup client connect to rank %s with worker socket path %s",
                rank,
                worker_socket_path,
            )

            push_socket = get_zmq_socket(
                self.ctx,
                worker_socket_path,
                "ipc",
                zmq.PUSH,  # type: ignore[attr-defined]
                "connect",
            )

            self.push_sockets.append(push_socket)

        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )
        logger.info(
            "lmcache lookup client connect to scheduler with socket path %s",
            scheduler_socket_path,
        )

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.token_database: TokenDatabase
        if config.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        # A lock is needed since we need another thread to pull
        # responses from the lookup_and_prefetch server
        # (e.g., worker process).
        self.lock = threading.RLock()

        # map from lookup_id (i.e., req_id) to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = {}

        # map from lookup_id (i.e., req_id) to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}

        # The two parts are [lookup_id (i.e., req_id), num_hit_tokens]
        self.num_parts = 2

        # Track lookup_ids that have been aborted for cleanup
        self.aborted_lookups: set[str] = set()
        self.cleanup_res_for_each_worker: dict[str, set[int]] = {}

        self.running = True

        self.thread = threading.Thread(
            target=self.process_responses_from_workers,
            daemon=True,
            name="async-lookup-client-thread",
        )
        self.thread.start()

        # default backoff time
        self.lookup_backoff_time = 0.01
        if config.extra_config is not None:
            self.lookup_backoff_time = float(
                config.extra_config.get("lookup_backoff_time", self.lookup_backoff_time)
            )

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        -1 means not found;
        None means ongoing;
        int >= 0 means number of hit tokens
        """
        with self.lock:
            # Do not reuse a request ID until every response from its cancelled
            # lookup has arrived. The worker defers cleanup until that response.
            if lookup_id in self.aborted_lookups:
                return None
            if (req_status := self.reqs_status.get(lookup_id, -1)) == -1:
                self.reqs_status[lookup_id] = None
                self.first_lookup_time[lookup_id] = time.time()
            elif req_status is None:
                time.sleep(self.lookup_backoff_time)
                if (
                    time.time() - self.first_lookup_time[lookup_id]
                ) * 1000 > self.config.lookup_timeout_ms:
                    logger.warning(
                        (
                            "Request %s is still waiting for async lookup "
                            "after %d seconds, returning 0 lmcache cached tokens "
                            "so vllm can recompute"
                        ),
                        lookup_id,
                        self.config.lookup_timeout_ms // 1000,
                    )
                    self.cancel_lookup(lookup_id)
                    self.first_lookup_time.pop(lookup_id, None)
                    return 0

            return req_status

    # TODO(Jiayi): Consider batching here
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        hashes: list[int] = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)  # type: ignore[arg-type]
            offsets.append(end - start)

        # Create structured message
        msg = LookupRequestMsg(
            lookup_id=lookup_id,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        )

        # Serialize message using msgspec
        msg_buf = msgspec.msgpack.encode(msg)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    def process_responses_from_workers(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                # Deserialize message using msgspec
                msg = msgspec.msgpack.decode(
                    msg_buf,
                    type=Union[LookupResponseMsg, LookupCleanupResponseMsg],
                )
                lookup_id = msg.lookup_id

                with self.lock:
                    if isinstance(msg, LookupCleanupResponseMsg):
                        workers = self.cleanup_res_for_each_worker.setdefault(
                            lookup_id, set()
                        )
                        workers.add(msg.worker_id)
                        if len(workers) == self.world_size:
                            self.cleanup_res_for_each_worker.pop(lookup_id, None)
                            self.res_for_each_worker.pop(lookup_id, None)
                            self.reqs_status.pop(lookup_id, None)
                            self.first_lookup_time.pop(lookup_id, None)
                            self.aborted_lookups.discard(lookup_id)
                        continue

                    res = msg.num_hit_tokens
                    if lookup_id not in self.res_for_each_worker:
                        self.res_for_each_worker[lookup_id] = [res]
                    else:
                        self.res_for_each_worker[lookup_id].append(res)
                    all_res = self.res_for_each_worker[lookup_id]

                    if len(all_res) == self.world_size:
                        self.res_for_each_worker.pop(lookup_id)

                        min_hit = min(all_res)
                        max_hit = max(all_res)
                        if min_hit != max_hit:
                            logger.warning(
                                "Lookup hit count differs across TP ranks for "
                                "req=%s: per_rank=%s min=%d max=%d. "
                                "Scheduler uses min; ranks above min may still "
                                "retrieve less than rank0 reported (garbage risk).",
                                lookup_id,
                                all_res,
                                min_hit,
                                max_hit,
                            )

                        if lookup_id in self.aborted_lookups:
                            # Do not resurrect scheduler state. The tombstone is
                            # cleared only after every worker acknowledges cleanup.
                            self.first_lookup_time.pop(lookup_id, None)
                        else:
                            # NOTE: it is possible that the number of hit
                            # tokens is different across (TP and PP) ranks, so
                            # use the minimum number of hit tokens.
                            self.reqs_status[lookup_id] = min_hit

            except Exception as e:
                logger.error("Error processing response from worker: %s", e)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self.lock:
            self.reqs_status.pop(lookup_id, None)
            self.first_lookup_time.pop(lookup_id, None)

    def cleanup_lookup(self, lookup_id: str) -> None:
        """Queue ordered worker cleanup and discard scheduler-side status."""
        with self.lock:
            already_pending = lookup_id in self.aborted_lookups
            self.aborted_lookups.add(lookup_id)
            self.reqs_status.pop(lookup_id, None)
            self.first_lookup_time.pop(lookup_id, None)
        if not already_pending:
            self._send_cleanup_message(lookup_id)

    def cancel_lookup(self, lookup_id: str) -> None:
        """Cancel a lookup and release worker state after terminal response."""
        self.cleanup_lookup(lookup_id)

    def _send_cleanup_message(self, lookup_id: str) -> None:
        """Send cleanup message to workers to release memory objects."""
        msg = LookupCleanupMsg(lookup_id=lookup_id)
        msg_buf = msgspec.msgpack.encode(msg)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        logger.debug("Sent cleanup message for lookup_id=%s", lookup_id)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)


class LMCacheAsyncLookupServer:
    """ZMQ-based async lookup server that handles lookup and prefetch
    requests using LMCacheEngine."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        worker_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_worker", rpc_port, metadata.worker_id
        )
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.push_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PUSH,  # type: ignore[attr-defined]
            "connect",
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            worker_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.worker_id = int(metadata.worker_id)
        self.running = True
        self._cleanup_lock = threading.Lock()
        # Request processing only submits the lookup coroutine. Cleanup arriving
        # next on the FIFO ZMQ pipe must therefore wait for the terminal response,
        # not merely for submission ordering.
        self._cleanup_state: dict[str, str] = {}

        logger.info(
            "lmcache lookup server start with"
            " scheduler socket path %s, "
            "worker socket path %s",
            scheduler_socket_path,
            worker_socket_path,
        )
        self.thread = threading.Thread(
            target=self.process_requests_from_scheduler,
            daemon=True,
            name="async-lookup-server-thread",
        )
        self.thread.start()

    def process_requests_from_scheduler(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                # rely on msgspec to automatically discriminate
                # between LookupRequestMsg and LookupCleanupMsg
                msg = msgspec.msgpack.decode(
                    msg_buf,
                    type=Union[LookupRequestMsg, LookupCleanupMsg],
                )

                if isinstance(msg, LookupRequestMsg):
                    # Handle lookup request
                    with self._cleanup_lock:
                        self._cleanup_state[msg.lookup_id] = "pending"
                    try:
                        self.lmcache_engine.async_lookup_and_prefetch(
                            lookup_id=msg.lookup_id,
                            hashes=msg.hashes,
                            offsets=msg.offsets,
                            pin=True,
                            request_configs=msg.request_configs,
                        )
                    except BaseException:
                        # Synchronous setup failed before a storage-loop lookup
                        # could own the terminal response. Complete the same
                        # response/cleanup handshake with a conservative miss.
                        self.send_response_to_scheduler(msg.lookup_id, 0)
                        raise

                elif isinstance(msg, LookupCleanupMsg):
                    self._request_cleanup(msg.lookup_id)

                else:
                    logger.warning("Unknown message type: %s", type(msg))

            except Exception as e:
                logger.error("Error processing request from scheduler: %s", e)

    def send_response_to_scheduler(
        self, lookup_id: str, num_hit_tokens: int
    ) -> None:
        """Send responses from the storage loop that owns the PUSH socket."""
        loop = self.lmcache_engine.storage_manager.loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not loop:
            loop.call_soon_threadsafe(
                self._send_response_to_scheduler,
                lookup_id,
                num_hit_tokens,
            )
            return
        self._send_response_to_scheduler(lookup_id, num_hit_tokens)

    def _send_response_to_scheduler(
        self, lookup_id: str, num_hit_tokens: int
    ) -> None:
        # Create structured response message
        msg = LookupResponseMsg(
            lookup_id=lookup_id,
            num_hit_tokens=num_hit_tokens,
        )

        # Serialize message using msgspec
        cleanup = False
        with self._cleanup_lock:
            if self._cleanup_state.get(lookup_id) == "cleanup_requested":
                self._cleanup_state.pop(lookup_id, None)
                cleanup = True
            else:
                self._cleanup_state.pop(lookup_id, None)
        try:
            msg_buf = msgspec.msgpack.encode(msg)
            self.push_socket.send(msg_buf, copy=False)
            if cleanup:
                # Preserve response-before-cleanup-ack ordering on this worker's
                # PUSH pipe. The client retains its tombstone until the ack.
                self._complete_cleanup(lookup_id)
        except BaseException:
            logger.exception(
                "Failed to send async lookup terminal response: lookup_id=%s",
                lookup_id,
            )
            raise

    def _request_cleanup(self, lookup_id: str) -> None:
        cleanup = False
        with self._cleanup_lock:
            state = self._cleanup_state.get(lookup_id)
            if state in ("pending", "cleanup_requested"):
                self._cleanup_state[lookup_id] = "cleanup_requested"
            else:
                self._cleanup_state.pop(lookup_id, None)
                cleanup = True
        if cleanup:
            loop = self.lmcache_engine.storage_manager.loop
            loop.call_soon_threadsafe(self._complete_cleanup, lookup_id)

    def _complete_cleanup(self, lookup_id: str) -> None:
        """Release lookup ownership and acknowledge from the response thread."""
        self.lmcache_engine.cleanup_memory_objs(lookup_id)
        msg = LookupCleanupResponseMsg(
            lookup_id=lookup_id,
            worker_id=self.worker_id,
        )
        self.push_socket.send(msgspec.msgpack.encode(msg), copy=False)

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            self.push_socket.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)
