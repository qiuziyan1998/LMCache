# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
from multiprocessing.connection import Client, Connection, Listener
from typing import Any, Sequence
import base64
import json
import os
import subprocess
import sys
import threading
import traceback

# First Party
from lmcache.logging import init_logger
from lmcache.v1.shared_cpu_cache import SharedSlabMapping
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)

logger = init_logger(__name__)


def _run_mooncake_process_worker(
    connection: Connection,
    config_dict: dict[str, Any],
    shm_name: str,
    slab_size: int,
    generation: int,
) -> None:
    '''Own the blocking Mooncake client outside the vLLM worker process.'''
    mapping: SharedSlabMapping | None = None
    store: Any = None
    registered = False
    try:
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        config = MooncakeStoreConfig(**config_dict)
        if config.storage_root_dir:
            os.environ['MOONCAKE_STORAGE_ROOT_DIR'] = config.storage_root_dir
        mapping = SharedSlabMapping.attach(
            shm_name=shm_name,
            size=slab_size,
            generation=generation,
            writable=True,
        )
        store = MooncakeDistributedStore()
        store.setup(
            config.local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )
        register_status = int(store.register_buffer(mapping.ptr, mapping.size))
        if register_status != 0:
            raise RuntimeError(
                'Mooncake process failed to register the shared CPU slab: '
                f'status={register_status}, shm_name={shm_name}, '
                f'slab_size={slab_size}'
            )
        registered = True
        connection.send(('ready', os.getpid()))

        while True:
            message = connection.recv()
            operation = message[0]
            if operation == 'close':
                connection.send(('closed', None))
                break
            if operation != 'get_pages':
                raise ValueError(
                    f'Unknown Mooncake process operation {operation!r}'
                )
            _, page_keys, page_offsets, page_sizes = message
            buffer_ptrs = [
                [mapping.ptr + int(offset) for offset in offsets]
                for offsets in page_offsets
            ]
            statuses = store.batch_get_into_multi_buffers(
                page_keys,
                buffer_ptrs,
                page_sizes,
            )
            connection.send(('result', [int(status) for status in statuses]))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except BaseException:
            pass
    finally:
        if store is not None:
            if registered and mapping is not None:
                try:
                    store.unregister_buffer(mapping.ptr)
                except BaseException:
                    pass
            try:
                store.close()
            except BaseException:
                pass
        if mapping is not None:
            try:
                mapping.close()
            except BaseException:
                pass
        connection.close()


class MooncakeProcessTransferClient:
    '''Synchronous IPC client for process-isolated page-first gets.

    The caller retains ownership of allocator objects. The child only attaches
    the same shared slab and writes into byte ranges already reserved by the
    caller, so allocator metadata never crosses the process boundary.
    '''

    def __init__(
        self,
        *,
        config: MooncakeStoreConfig,
        shm_name: str,
        slab_size: int,
        generation: int,
        startup_timeout: float,
    ) -> None:
        authkey = os.urandom(32)
        listener = Listener(
            ('127.0.0.1', 0),
            family='AF_INET',
            authkey=authkey,
        )
        raw_listener = getattr(listener, '_listener', None)
        raw_socket = getattr(raw_listener, '_socket', None)
        if raw_socket is not None:
            raw_socket.settimeout(startup_timeout)
        payload = {
            'address': list(listener.address),
            'authkey': base64.b64encode(authkey).decode('ascii'),
            'config': asdict(config),
            'shm_name': shm_name,
            'slab_size': int(slab_size),
            'generation': int(generation),
        }
        child_env = os.environ.copy()
        child_env['LMCACHE_MOONCAKE_PROCESS_PAYLOAD'] = json.dumps(payload)
        process = subprocess.Popen(
            [sys.executable, '-m', __name__],
            env=child_env,
        )
        try:
            self._connection = listener.accept()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            raise
        finally:
            listener.close()
        self._process = process
        self._lock = threading.Lock()
        self._closed = False

        try:
            status, payload = self._receive(startup_timeout, 'startup')
        except BaseException:
            self._terminate()
            raise
        if status != 'ready':
            self.close()
            raise RuntimeError(
                'Mooncake process isolation startup failed: '
                f'status={status}, details={payload}'
            )
        logger.info(
            'Mooncake cold-transfer process ready: pid=%s, shm_name=%s, '
            'slab_size=%s',
            payload,
            shm_name,
            slab_size,
        )

    def _receive(self, timeout: float, operation: str) -> tuple[str, Any]:
        if not self._connection.poll(timeout):
            raise TimeoutError(
                f'Mooncake process {operation} timed out after {timeout}s'
            )
        response = self._connection.recv()
        if not isinstance(response, tuple) or len(response) != 2:
            raise RuntimeError(
                f'Mooncake process returned invalid {operation} response: '
                f'{response!r}'
            )
        status, payload = response
        if status == 'error':
            raise RuntimeError(
                f'Mooncake process {operation} failed:\n{payload}'
            )
        return str(status), payload

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    def get_pages(
        self,
        page_keys: Sequence[str],
        page_offsets: Sequence[Sequence[int]],
        page_sizes: Sequence[Sequence[int]],
        *,
        timeout: float,
    ) -> list[int]:
        if self._closed:
            raise RuntimeError('Mooncake process transfer client is closed')
        if not (
            len(page_keys) == len(page_offsets) == len(page_sizes)
        ):
            raise ValueError(
                'Mooncake process page layout lengths differ: '
                f'keys={len(page_keys)}, offsets={len(page_offsets)}, '
                f'sizes={len(page_sizes)}'
            )
        with self._lock:
            try:
                self._connection.send(
                    (
                        'get_pages',
                        list(page_keys),
                        [list(offsets) for offsets in page_offsets],
                        [list(sizes) for sizes in page_sizes],
                    )
                )
                status, payload = self._receive(timeout, 'get_pages')
            except BaseException:
                self._terminate()
                raise
        if status != 'result':
            raise RuntimeError(
                'Mooncake process returned unexpected get_pages status '
                f'{status!r}'
            )
        return [int(value) for value in payload]

    def _terminate(self) -> None:
        self._closed = True
        self._connection.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                self._connection.send(('close',))
                try:
                    self._receive(5.0, 'close')
                except BaseException:
                    logger.warning(
                        'Mooncake cold-transfer process did not close cleanly'
                    )
        finally:
            self._connection.close()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)


def _main() -> None:
    raw_payload = os.environ.pop(
        'LMCACHE_MOONCAKE_PROCESS_PAYLOAD',
        None,
    )
    if raw_payload is None:
        raise RuntimeError('Mooncake process payload is missing')
    payload = json.loads(raw_payload)
    connection = Client(
        tuple(payload['address']),
        family='AF_INET',
        authkey=base64.b64decode(payload['authkey']),
    )
    _run_mooncake_process_worker(
        connection,
        payload['config'],
        payload['shm_name'],
        int(payload['slab_size']),
        int(payload['generation']),
    )


if __name__ == '__main__':
    _main()
