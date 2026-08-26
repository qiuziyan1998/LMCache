# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, List, Optional
import abc
import asyncio

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.protocol import get_remote_metadata_bytes, init_remote_metadata_info

logger = init_logger(__name__)


def resolve_save_chunk_meta(config: LMCacheEngineConfig) -> bool:
    """
    Whether remote connectors embed per-chunk metadata alongside KV bytes.

    Explicit ``save_chunk_meta`` in extra_config always wins. Otherwise layerwise
    mode defaults to True for legacy remote backends that cannot rely on fixed
    connector-side metadata.
    """
    extra = config.extra_config
    if extra is not None and "save_chunk_meta" in extra:
        return bool(extra["save_chunk_meta"])
    if extra is None:
        return True
    if config.use_layerwise:
        return True
    return bool(extra.get("save_chunk_meta", True))


def layerwise_connector_meta_shapes(
    shapes: list[torch.Size],
) -> list[torch.Size]:
    """Per-layer keys store one layer per chunk; shrink the layer dimension."""
    adjusted: list[torch.Size] = []
    for shape in shapes:
        if len(shape) == 4 and shape[1] > 1:
            adjusted.append(torch.Size([shape[0], 1, shape[2], shape[3]]))
        else:
            adjusted.append(shape)
    return adjusted


def NotAudit(func):
    """
    Decorator to mark methods that should not be audited.
    These methods will be directly forwarded to the real connector without logging.
    """
    func._not_audit = True
    return func


class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """

    def __init__(
        self, config: LMCacheEngineConfig, metadata: Optional[LMCacheMetadata]
    ):
        """
        Initialize some common attributes, which will be used in the subclasses.
        - `save_chunk_meta` is a flag to indicate whether to save the chunk meta info.
        - `meta_shapes` is a list of shapes of lmcache full chunk.
        - `meta_dtypes` is a list of dtypes of lmcache chunk.
        - `meta_fmt` is the memory format of the lmcache chunk.
        - `full_chunk_size_bytes` is the size of the lmcache full chunk.
        - `single_token_size` is the size of a single token.`
        - `remote_metadata_bytes` is the size of the remote metadata.

        Input:
            config: the lmcache engine config
            metadata: the lmcache engine metadata
        """
        # TODO(chunxiaozheng): support layerwise here
        assert metadata is not None
        self.save_chunk_meta: bool = resolve_save_chunk_meta(config)
        self.meta_shapes: list[torch.Size] = metadata.get_shapes()
        if config.use_layerwise and not self.save_chunk_meta:
            self.meta_shapes = layerwise_connector_meta_shapes(self.meta_shapes)
        self.meta_dtypes: list[torch.dtype] = metadata.get_dtypes()
        self.meta_fmt: MemoryFormat = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )
        self.full_chunk_size_bytes: int = get_size_bytes(
            self.meta_shapes, self.meta_dtypes
        )
        assert self.full_chunk_size_bytes % metadata.chunk_size == 0
        self.single_token_size = self.full_chunk_size_bytes // metadata.chunk_size

        # init remote metadata info
        init_remote_metadata_info(metadata.get_num_groups())
        self.remote_metadata_bytes = get_remote_metadata_bytes()
        logger.info(
            "init remote connector metadata info, shapes: %s, dtypes: %s, fmt: %s, "
            "full chunk size: %s, single token size: %s, remote metadata bytes: %s",
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
            self.full_chunk_size_bytes,
            self.single_token_size,
            self.remote_metadata_bytes,
        )

    @NotAudit
    def reshape_partial_chunk(
        self,
        memory_obj: MemoryObj,
        bytes_read: int,
    ) -> MemoryObj:
        assert self.full_chunk_size_bytes is not None
        assert self.single_token_size is not None
        if (
            bytes_read == 0
            or bytes_read % self.single_token_size != 0
            or bytes_read > self.full_chunk_size_bytes
        ):
            raise ValueError(
                f"bytes_read: {bytes_read} is illegal, "
                f"single_token_size: {self.single_token_size}, "
                f"full_chunk_size_bytes: {self.full_chunk_size_bytes}"
            )

        if bytes_read == self.full_chunk_size_bytes:
            # full chunk, return directly
            return memory_obj

        # NOTE: for unfull chunk, we have no way to verify
        shape_list = list(memory_obj.meta.shape)
        shape_list[2] = bytes_read // self.single_token_size
        actual_shape = torch.Size(shape_list)
        resize_raw_view = getattr(memory_obj, "resize_raw_view", None)
        if callable(resize_raw_view):
            resize_raw_view(bytes_read)
        else:
            memory_obj.raw_data = memory_obj.raw_data[:bytes_read]
        memory_obj.meta.shape = actual_shape
        if memory_obj.meta.shapes:
            memory_obj.meta.shapes = [actual_shape]
        refresh_metadata_view = getattr(memory_obj, "refresh_metadata_view", None)
        if callable(refresh_metadata_view):
            refresh_metadata_view()

        return memory_obj

    @NotAudit
    def post_init(self):
        """
        Post-initialization method to be called after the connector is created.
        This can be used to perform any additional setup required by the connector.
        """
        logger.info("Dummy post-initializing remote connector")

    @abc.abstractmethod
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a CacheEngineKey

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists_sync(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key synchronized

        Input:
            key: a CacheEngineKey

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Get the memory_obj of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The memory_obj of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Send the memory_obj with the corresponding key directly
        to the remote server. Will decrease the ref count after
        send finishes.

        Input:
            key: the CacheEngine key
            memory_obj: the memory_obj of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        """
        Close remote server

        """
        raise NotImplementedError

    def support_ping(self) -> bool:
        """
        Check if the connector supports ping operation

        Returns:
            True if ping is supported, False otherwise
        """
        return False

    async def ping(self) -> int:
        """
        Ping the remote server

        Returns:
            The error code, 0 means success
        """
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return False

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batched get the memory_objs of the corresponding keys

        Input:
            keys: the keys of the corresponding objects

        Returns:
            The memory_objs of the corresponding keys
            Return None if the key does not exist
        """
        raise NotImplementedError

    def support_batched_put(self) -> bool:
        """
        Check if the connector supports batched put
        Returns:
            True if batched put is supported, False otherwise
        """
        return False

    def requires_put_completion(self) -> bool:
        """Whether callers must wait for remote persistence."""
        return False

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """
        Batched put the memory_objs with the corresponding keys
        Input:
            keys: the keys of the corresponding objects
            memory_objs: the memory_objs of the corresponding keys
        """
        raise NotImplementedError

    async def batched_put_external_pages(
        self,
        keys: List[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        producer_events: tuple[Any, ...] | Any,
        req_id: str,
    ) -> None:
        """Persist externally owned buffers in the connector's page format."""
        raise NotImplementedError

    async def batched_get_external_pages(
        self,
        keys: List[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        req_id: str,
    ) -> None:
        """Retrieve page values directly into externally owned buffers."""
        raise NotImplementedError

    async def push_external_pages(
        self,
        *,
        remote_session: str,
        source_plan: Any,
        destination_descriptors: tuple[Any, ...],
        activation: Any,
    ) -> Any:
        """Push registered source pages into armed remote destinations."""
        raise NotImplementedError

    async def prepare_remote_fill_source(self, source_plan: Any) -> Any:
        """Fence and register direct-push sources before destination ARM."""
        raise NotImplementedError

    def get_remote_fill_destination_session(self) -> str | None:
        """Return this process's registered native destination session."""
        return None

    def batched_external_pages_exist(
        self, keys: List[CacheEngineKey]
    ) -> List[bool]:
        """Check arbitrary page keys in one native operation."""
        raise NotImplementedError

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check how many keys exist in file system in batch

        Args:
            lookup_id: Identifier for this lookup operation
            keys: List of keys to check
            pin: Whether to pin the keys (not used in FS connector)

        Returns:
            Number of consecutive keys that exist, starting from the first key
        """
        tasks = [self.exists(key) for key in keys]
        results = await asyncio.gather(*tasks)
        if False in results:
            return results.index(False)
        return len(results)

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Batched get the memory_objs of the corresponding keys (non-blocking)

        This method returns only the consecutive prefix of successfully retrieved
        memory objects. Once a key is not found (None) or an exception occurs,
        all subsequent memory objects (even if successfully retrieved) will be
        released to avoid memory leaks, and only the prefix before the first
        failure will be returned.

        Args:
            lookup_id: Identifier for this lookup operation
            keys: List of keys to get

        Returns:
            List of consecutive memory objects from the beginning until the first
            failure (None or Exception). Empty list if the first key fails.
        """
        # Use asyncio.gather to fetch all keys concurrently
        results = await asyncio.gather(
            *(self.get(key) for key in keys), return_exceptions=True
        )

        # Only return consecutive prefix of valid memory objects
        memory_objs = []
        found_failure = False
        for result in results:
            if found_failure:
                # Release subsequent memory objects to avoid memory leak
                if isinstance(result, MemoryObj):
                    result.ref_count_down()
            elif isinstance(result, MemoryObj):
                memory_objs.append(result)
            else:
                # First failure encountered (None or Exception)
                if isinstance(result, Exception):
                    logger.warning(f"Exception during batched get: {result}")
                found_failure = True

        return memory_objs

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """
        Remove a memory object.

        :param CacheEngineKey key: The key of the MemoryObj.

        :return: a bool indicates whether remove is successful.
        """
        raise NotImplementedError

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """
        Batched contains.

        :param List[CacheEngineKey] keys: The keys to check.

        :return: Return hit chunks by prefix match.
        """
        raise NotImplementedError

    def support_batched_contains(self) -> bool:
        """
        Is supported batched_contains
        """
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
