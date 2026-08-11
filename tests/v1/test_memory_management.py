# SPDX-License-Identifier: Apache-2.0
# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    GPUMemoryAllocator,
    HostMemoryAllocator,
    LayerPageMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.pin_monitor import PinMonitor


def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    shape1 = torch.Size([512, 512])
    data1 = allocator.allocate(shape1, torch.float)
    assert data1 is not None
    assert data1.tensor.dtype == torch.float
    assert data1.tensor.shape == shape1

    # 1024 * 1024 * 2 = 2MB
    shape2 = torch.Size([1024, 1024])
    data2 = allocator.allocate(shape2, torch.bfloat16)
    assert data2 is not None
    assert data2.tensor.dtype == torch.bfloat16
    assert data2.tensor.shape == shape2

    # 2048 * 2048 * 1 = 4MB
    shape3 = torch.Size([2048, 2048])
    data3 = allocator.allocate(shape3, torch.int8)
    assert data3 is not None
    assert data3.tensor.dtype == torch.int8
    assert data3.tensor.shape == shape3

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    shape4 = torch.Size([3, 5, 7])
    data4 = allocator.allocate(shape4, torch.half)
    assert data4 is not None
    assert data4.tensor.dtype == torch.half
    assert data4.tensor.shape == shape4

    data_fail = allocator.allocate(
        torch.Size([max_size]), torch.float
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()

    allocator.close()


def check_paged_allocator(allocator, shape, dtype, fmt, max_num_pages):
    # Allocate one page
    data1 = allocator.allocate(shape, dtype, fmt)
    assert data1 is not None
    assert data1.tensor.dtype == dtype
    assert data1.tensor.shape == shape

    # Allocate another 2 pages
    data2 = allocator.batched_allocate(shape, dtype, 2, fmt)

    for data in data2:
        assert data is not None
        assert data.tensor.dtype == dtype
        assert data.tensor.shape == shape

    # Allocate a smaller page
    smaller_shape = torch.Size([2, 32, 8, 1024])
    data3 = allocator.allocate(smaller_shape, dtype, fmt)
    assert data3 is not None
    assert data3.tensor.dtype == dtype
    assert data3.tensor.shape == smaller_shape

    allocator.free(data3)
    assert allocator.memcheck()

    allocator.batched_free(data2)
    assert allocator.memcheck()

    allocator.free(data1)
    assert allocator.memcheck()

    data_fail = allocator.batched_allocate(
        shape, dtype, max_num_pages + 1, fmt
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize(
    "use_paging",
    [True, False],
)
def test_tensor_allocator(use_paging):
    total_size = 1024 * 1024 * 128  # 128MB
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")
    if use_paging:
        shape = torch.Size([2, 32, 16, 1024])  # 64 pages
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD
        num_pages = 64
        allocator = PagedTensorMemoryAllocator(tensor_buffer, [shape], [dtype], fmt)
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        allocator = TensorMemoryAllocator(tensor_buffer)
        check_allocator(allocator, total_size)

    allocator.close()


def test_tensor_batched_allocate_contiguous_and_fragmented() -> None:
    block_size = 4096

    class TracingAllocator(TensorMemoryAllocator):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__(tensor)
            self.slice_calls: list[tuple[int, int]] = []

        def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
            self.slice_calls.append((start, size))
            return super()._get_buffer_slice(start, size)

    allocator = TracingAllocator(torch.zeros(block_size * 5, dtype=torch.uint8))
    assert allocator.batched_allocate(torch.Size([block_size]), torch.uint8, 0) == []
    assert allocator.slice_calls == []
    contiguous = allocator.batched_allocate(torch.Size([block_size]), torch.uint8, 3)
    assert contiguous is not None
    assert allocator.slice_calls == [(0, block_size * 3)]
    assert [obj.meta.address for obj in contiguous] == [0, block_size, block_size * 2]
    allocator.batched_free(contiguous)

    allocated = [
        allocator.allocate(torch.Size([block_size]), torch.uint8) for _ in range(5)
    ]
    allocated_objs = [obj for obj in allocated if obj is not None]
    assert len(allocated_objs) == 5
    allocator.free(allocated_objs[0])
    allocator.free(allocated_objs[2])
    allocator.slice_calls.clear()

    fragmented = allocator.batched_allocate(torch.Size([block_size]), torch.uint8, 2)
    assert fragmented is not None
    assert allocator.slice_calls == [(0, block_size), (block_size * 2, block_size)]

    allocator.batched_free(fragmented)
    allocator.batched_free([allocated_objs[index] for index in (1, 3, 4)])
    assert allocator.memcheck()


def test_tensor_batched_allocation_provenance_validates_tail_size() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(16384, dtype=torch.uint8))
    objects = allocator.batched_allocate(
        torch.Size([1024]),
        torch.uint8,
        3,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert objects is not None

    validation_cache = {}
    expected = {
        "parent": allocator,
        "slab_size": allocator.buffer.numel(),
        "dtype": torch.uint8,
        "fmt": MemoryFormat.KV_MLA_LATENT_FMT,
    }
    assert all(
        obj.is_trusted_allocation_batch_member(validation_cache, **expected)
        for obj in objects
    )
    assert len(validation_cache) == 1

    logical_size = objects[-1].get_size()
    objects[-1].group_prefix_sum = (0, objects[-1].meta.phy_size + 1)
    assert not objects[-1].is_trusted_allocation_batch_member(
        validation_cache, **expected
    )
    objects[-1].group_prefix_sum = (0, logical_size)
    objects[-1].meta.dtype = torch.float16
    assert not objects[-1].is_trusted_allocation_batch_member(
        validation_cache, **expected
    )
    objects[-1].meta.dtype = torch.uint8
    objects[-1].parent_allocator = None
    assert not objects[-1].is_trusted_allocation_batch_member(
        validation_cache, **expected
    )
    objects[-1].parent_allocator = allocator
    assert not objects[0].is_trusted_allocation_batch_member(
        {},
        **{**expected, "fmt": MemoryFormat.KV_DSA_INDEX_FMT},
    )

    allocator.batched_free(objects)
    assert not any(
        obj.is_trusted_allocation_batch_member(validation_cache, **expected)
        for obj in objects
    )


def test_tensor_address_backed_batch_preserves_tensor_access() -> None:
    block_size = 4096
    buffer = torch.zeros(block_size * 3, dtype=torch.uint8)
    allocator = TensorMemoryAllocator(buffer)
    objects = allocator.batched_allocate_address_backed(
        torch.Size([block_size]), torch.uint8, 3
    )
    assert objects is not None
    assert all(obj.has_tensor_storage for obj in objects)
    assert len({id(obj.group_prefix_sum) for obj in objects}) == 1
    assert [obj.data_ptr for obj in objects] == [
        buffer.data_ptr() + obj.meta.address for obj in objects
    ]

    objects[-1].resize_raw_view(1024)
    objects[-1].meta.shape = torch.Size([1024])
    objects[-1].meta.shapes = [torch.Size([1024])]
    objects[-1].refresh_metadata_view()
    assert objects[0].get_size() == block_size
    raw_tensor = objects[-1].raw_tensor
    assert raw_tensor is not None and raw_tensor.numel() == 1024
    assert objects[-1].raw_data is raw_tensor
    assert objects[-1].tensor is not None
    assert objects[-1].tensor.shape == torch.Size([1024])
    with pytest.raises(ValueError, match="Invalid raw tensor view size"):
        objects[-1].resize_raw_view(2048)

    allocator.batched_free(objects)
    assert allocator.memcheck()


def test_layer_page_batch_owns_one_object_per_chunk() -> None:
    shape = torch.Size([8, 4])
    layer_bytes = shape.numel() * torch.bfloat16.itemsize
    buffer = torch.zeros(layer_bytes * 3 * 2 + 8192, dtype=torch.uint8)
    allocator = TensorMemoryAllocator(buffer)
    pages = allocator.batched_allocate_layer_pages(
        shape,
        torch.bfloat16,
        batch_size=2,
        num_layers=3,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )

    assert pages is not None and len(pages) == 2
    assert all(isinstance(page, LayerPageMemoryObj) for page in pages)
    assert all(
        page.num_layers == 3 and page.layer_size == layer_bytes for page in pages
    )
    for page in pages:
        tensors = [page.layer_tensor(layer) for layer in range(3)]
        assert all(tensor.shape == shape for tensor in tensors)
        assert [tensor.data_ptr() for tensor in tensors] == [
            page.data_ptr + layer * layer_bytes for layer in range(3)
        ]
        for layer, tensor in enumerate(tensors):
            tensor.fill_(layer + 1)
        assert [int(tensor.flatten()[0]) for tensor in tensors] == [1, 2, 3]

    allocator.batched_free(pages)
    assert allocator.memcheck()


@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
def test_layer_pages_use_exact_partial_shapes_and_bytes(chunk_size: int) -> None:
    valid_counts = sorted(
        {
            count
            for count in (1, 255, 256, 257, 511, 512, 513, chunk_size - 1)
            if count <= chunk_size
        }
    )
    shape = torch.Size([chunk_size, 4])
    allocator = TensorMemoryAllocator(
        torch.zeros(
            chunk_size * 4 * 2 * 2 * len(valid_counts) + 65536,
            dtype=torch.uint8,
        )
    )
    pages = allocator.batched_allocate_layer_pages(
        shape,
        torch.float16,
        batch_size=len(valid_counts),
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        valid_tokens=valid_counts,
    )

    assert pages is not None
    for page, valid_tokens in zip(pages, valid_counts, strict=True):
        expected_layer_bytes = valid_tokens * 4 * torch.float16.itemsize
        assert page.valid_tokens == valid_tokens
        assert page.metadata.valid_tokens == valid_tokens
        assert page.layer_size == expected_layer_bytes
        assert page.get_size() == expected_layer_bytes * 2
        assert page.get_shapes() == [torch.Size([valid_tokens, 4])] * 2
        assert all(
            page.layer_tensor(layer).shape == torch.Size([valid_tokens, 4])
            for layer in range(2)
        )

    allocator.batched_free(pages)
    assert allocator.memcheck()


def test_partial_layer_pages_batch_equal_token_counts_together() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(65536, dtype=torch.uint8))
    calls = []
    allocate = allocator.address_manager.batched_allocate

    def recording_allocate(size, batch_size):
        calls.append(batch_size)
        return allocate(size, batch_size)

    allocator.address_manager.batched_allocate = recording_allocate
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8, 4]),
        torch.float16,
        batch_size=3,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        valid_tokens=[8, 3, 8],
    )

    assert pages is not None
    assert calls == [2, 1]
    assert [page.valid_tokens for page in pages] == [8, 3, 8]
    assert [page.layer_size for page in pages] == [64, 24, 64]
    allocator.batched_free(pages)
    assert allocator.memcheck()


@pytest.mark.parametrize(
    "fmt,width",
    (
        (MemoryFormat.KV_MLA_LATENT_FMT, 9),
        (MemoryFormat.KV_DSA_INDEX_FMT, 3),
    ),
)
def test_flat_layer_pages_preserve_full_and_tail_byte_layout(fmt, width) -> None:
    allocator = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([4 * width]),
        torch.float16,
        batch_size=2,
        num_layers=2,
        fmt=fmt,
        valid_tokens=[4, 3],
        full_tokens=4,
    )

    assert pages is not None
    for page, tokens in zip(pages, (4, 3), strict=True):
        values = torch.arange(tokens * width, dtype=torch.float16)
        assert page.get_shape() == torch.Size([tokens * width])
        assert page.layer_size == values.numel() * values.element_size()
        for layer in range(2):
            page.layer_tensor(layer).copy_(values)
            assert torch.equal(page.layer_tensor(layer), values)
    allocator.batched_free(pages)
    assert allocator.memcheck()


def test_flat_layer_pages_require_explicit_full_token_count() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))

    with pytest.raises(ValueError, match="requires full_tokens"):
        allocator.batched_allocate_layer_pages(
            torch.Size([4 * 9]),
            torch.float16,
            batch_size=1,
            num_layers=2,
            fmt=MemoryFormat.KV_MLA_LATENT_FMT,
            valid_tokens=3,
        )


def test_partial_layer_page_batch_rolls_back_on_allocation_failure() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))

    pages = allocator.batched_allocate_layer_pages(
        torch.Size([256, 4]),
        torch.float16,
        batch_size=2,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        valid_tokens=[1, 255],
    )

    assert pages is None
    assert allocator.memcheck()
    assert allocator.total_allocated_size == 0


def test_layer_page_rejects_pointer_access_after_release() -> None:
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        full_tokens=8,
    )
    assert pages is not None
    page = pages[0]

    page.ref_count_down()

    with pytest.raises(RuntimeError, match="no longer valid"):
        page.layer_data_ptr(0)
    with pytest.raises(RuntimeError, match="no longer valid"):
        page.layer_tensor(0)


def test_mixed_address_backed_batch_preserves_format_validation() -> None:
    allocator = object.__new__(MixedMemoryAllocator)
    allocator.pin_allocator = TensorMemoryAllocator(
        torch.zeros(4096, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="Unsupported memory format"):
        allocator.batched_allocate_address_backed(
            torch.Size([1024]), torch.uint8, 1, MemoryFormat.UNDEFINED
        )


def test_paged_allocator_refreshes_size_for_reused_partial_pages():
    full_shape = torch.Size([32, 8])
    partial_shape = torch.Size([19, 8])
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_T2D
    full_bytes = full_shape.numel() * dtype.itemsize
    partial_bytes = partial_shape.numel() * dtype.itemsize
    tensor_buffer = torch.zeros(full_bytes * 2, dtype=torch.uint8, device="cpu")

    allocator = PagedTensorMemoryAllocator(tensor_buffer, [full_shape], [dtype], fmt)

    full = allocator.allocate(full_shape, dtype, fmt)
    assert full is not None
    assert full.get_size() == full_bytes
    allocator.free(full)

    partial = allocator.allocate(partial_shape, dtype, fmt)
    assert partial is not None
    assert partial.get_shape() == partial_shape
    assert partial.get_size() == partial_bytes
    assert partial.raw_data.numel() == partial_bytes
    allocator.free(partial)

    batched = allocator.batched_allocate(partial_shape, dtype, 1, fmt)
    assert batched is not None
    assert batched[0].get_shape() == partial_shape
    assert batched[0].get_size() == partial_bytes
    assert batched[0].raw_data.numel() == partial_bytes
    allocator.batched_free(batched)

    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
@pytest.mark.parametrize(
    "use_paging",
    [
        False,
        True,
    ],
)
def test_device_allocators(alloc_cls, use_paging):
    total_size = 1024 * 1024 * 128  # 128MB

    shape = torch.Size([2, 32, 16, 1024])  # 64 pages
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_2LTD

    allocator = alloc_cls(
        total_size, use_paging=use_paging, shapes=[shape], dtypes=[dtype], fmt=fmt
    )

    if use_paging:
        num_pages = 64
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_inplace_modification(alloc_cls):
    total_size = 1024 * 1024
    allocator = alloc_cls(total_size)

    shape = torch.Size([4096])
    data = allocator.allocate(shape, torch.float)
    assert data is not None
    assert data.tensor.dtype == torch.float
    assert data.tensor.shape == shape

    data.tensor.fill_(1.0)
    assert torch.all(data.tensor == 1.0)

    data.tensor[1] = 2.0
    assert data.tensor[1] == 2.0

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_boundary_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)

    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, torch.float)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate(shape, torch.float)

    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_batched_alloc(alloc_cls):
    total_size = 32 * 100 * 2 * 1024 * 2
    batch_size = 32
    allocator = alloc_cls(total_size)
    shape = torch.Size([100, 2, 1024])
    objs = allocator.batched_allocate(
        shape, torch.bfloat16, batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == torch.bfloat16
        assert obj.tensor.shape == shape
    allocator.batched_free(objs)

    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        MixedMemoryAllocator,
    ],
)
def test_mixed_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    shape = torch.Size([512, 10])
    data1 = allocator.allocate(shape, [], MemoryFormat.BINARY_BUFFER)
    allocator.allocate(shape, torch.float)
    allocator.free(data1)

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    allocator.memcheck()
    allocator.close()


def test_memory_obj_metadata_to_and_from_dict():
    shape1 = torch.Size([128, 10])
    dtype1 = torch.float
    shape2 = torch.Size([256, 10])
    dtype2 = torch.uint8
    shapes = [shape1, shape2]
    dtypes = [dtype1, dtype2]
    metadata1 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
    )
    dict1 = metadata1.to_dict()
    metadata_from_dict_1 = MemoryObjMetadata.from_dict(dict1)
    assert metadata_from_dict_1.shape == shape1
    assert metadata_from_dict_1.dtype == dtype1
    assert metadata_from_dict_1.shapes is None
    assert metadata_from_dict_1.dtypes is None

    metadata2 = MemoryObjMetadata(
        shape=shape1,
        dtype=dtype1,
        address=0,
        phy_size=0,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
        shapes=shapes,
        dtypes=dtypes,
    )
    dict2 = metadata2.to_dict()
    metadata_from_dict_2 = MemoryObjMetadata.from_dict(dict2)
    assert metadata_from_dict_2.shape == shape1
    assert metadata_from_dict_2.dtype == dtype1
    assert metadata_from_dict_2.shapes == shapes
    assert metadata_from_dict_2.dtypes == dtypes


@pytest.mark.parametrize(
    "alloc_cls,custom_timeout,elapsed_time",
    [
        (HostMemoryAllocator, None, 360),
        (PinMemoryAllocator, None, 360),
        (GPUMemoryAllocator, None, 360),
        (MixedMemoryAllocator, None, 360),
        (HostMemoryAllocator, 60, 90),
    ],
)
def test_pin_timeout(alloc_cls, custom_timeout, elapsed_time):
    # Reset the singleton to ensure clean state
    LMCStatsMonitor.DestroyInstance()
    # Also reset the class variable to use the new singleton
    TensorMemoryObj.monitor = LMCStatsMonitor.GetOrCreate()

    # Reset and initialize PinMonitor
    PinMonitor._instance = None
    config = LMCacheEngineConfig.from_defaults()
    PinMonitor.GetOrCreate(config)

    try:
        total_size = 1024 * 1024
        allocator = alloc_cls(total_size)

        # Create a memory object
        data = allocator.allocate(torch.Size([4096]), torch.float)
        assert data is not None

        # Pin the object
        data.pin()
        assert data.metadata.pin_count == 1

        # Get initial forced unpin count
        monitor = LMCStatsMonitor.GetOrCreate()
        initial_forced_unpin_count = monitor.interval_forced_unpin_count

        # Get the PinMonitor instance that was used by pin()
        pin_monitor = PinMonitor.GetOrCreate()

        # Override timeout if custom timeout is specified
        if custom_timeout is not None:
            pin_monitor._pin_timeout_sec = custom_timeout

        # Simulate timeout by manually setting register time in PinMonitor
        obj_id = id(data)
        with pin_monitor._objects_lock:
            if obj_id in pin_monitor._pinned_objects:
                memory_obj, _ = pin_monitor._pinned_objects[obj_id]
                pin_monitor._pinned_objects[obj_id] = (
                    memory_obj,
                    time.time() - elapsed_time,
                )

        # Force a timeout check
        pin_monitor._check_timeouts()

        # Verify that pin_count is now 0
        assert data.metadata.pin_count == 0

        # Verify that forced unpin count increased
        assert monitor.interval_forced_unpin_count == initial_forced_unpin_count + 1

        allocator.close()
    finally:
        pass


def test_pin_monitor_timeout():
    """Test that PinMonitor correctly detects and handles pin timeouts."""

    # Create a mock memory object for testing
    class MockMemoryObjMetadata:
        def __init__(self):
            self.address = 12345
            self.pin_count = 0
            self.ref_count = 1

    class MockMemoryObj:
        def __init__(self):
            self.meta = MockMemoryObjMetadata()
            self.lock = threading.Lock()
            self.parent_allocator = None

        def unpin(self):
            self.meta.pin_count -= 1
            if self.meta.pin_count == 0:
                PinMonitor.GetOrCreate().on_unpin(self)
            if self.meta.pin_count < 0:
                self.meta.pin_count = 0

    # Reset PinMonitor singleton for testing
    PinMonitor._instance = None

    # Create PinMonitor with short timeout for testing
    config = LMCacheEngineConfig.from_defaults(
        pin_timeout_sec=1, pin_check_interval_sec=1
    )
    pin_monitor = PinMonitor.GetOrCreate(config)

    # Create a mock memory object
    mock_obj = MockMemoryObj()

    # Test registration
    pin_monitor.on_pin(mock_obj)
    assert pin_monitor.get_monitored_count() == 1

    # Test unregistration
    pin_monitor.on_unpin(mock_obj)
    assert pin_monitor.get_monitored_count() == 0

    # Test timeout detection
    try:
        # Register object first
        mock_obj.meta.pin_count = 1
        pin_monitor.on_pin(mock_obj)

        # Manually set old register time to simulate timeout
        # Set to 2 seconds ago to exceed the 1 second timeout
        obj_id = id(mock_obj)
        with pin_monitor._objects_lock:
            if obj_id in pin_monitor._pinned_objects:
                memory_obj, _ = pin_monitor._pinned_objects[obj_id]
                pin_monitor._pinned_objects[obj_id] = (
                    memory_obj,
                    time.time() - 2.0,
                )

        # Force a timeout check
        pin_monitor._check_timeouts()

        # Verify object was unpinned
        assert mock_obj.meta.pin_count == 0
        assert pin_monitor.get_monitored_count() == 0

    finally:
        pass


def test_pin_monitor_background_thread():
    """Test that PinMonitor background thread starts correctly."""
    # Reset singleton and create with config
    PinMonitor._instance = None
    config = LMCacheEngineConfig.from_defaults()
    pin_monitor = PinMonitor.GetOrCreate(config)

    # PinMonitor auto-starts in __init__, so it should already be running
    # PinMonitor now inherits from PeriodicThread, use is_running property
    assert pin_monitor.is_running
    assert pin_monitor._thread is not None
    assert pin_monitor._thread.is_alive()

    # Give thread a moment to start
    time.sleep(0.1)

    # Test basic functionality without stopping the thread
    # (thread stopping is handled by daemon thread behavior)


def test_tensor_memory_obj_pin_monitor_integration():
    """Test integration between TensorMemoryObj and PinMonitor."""

    # Create a simple allocator for testing
    class MockAllocator:
        def free(self, obj):
            pass

    # Create a real TensorMemoryObj
    raw_data = torch.empty(100, dtype=torch.float32)
    metadata = MemoryObjMetadata(
        shape=torch.Size([100]),
        dtype=torch.float32,
        address=12345,
        phy_size=400,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )

    allocator = MockAllocator()
    memory_obj = TensorMemoryObj(raw_data, metadata, allocator)

    # Get PinMonitor instance
    pin_monitor = PinMonitor.GetOrCreate()
    initial_count = pin_monitor.get_monitored_count()

    # Test pinning registers with PinMonitor
    memory_obj.pin()
    assert pin_monitor.get_monitored_count() == initial_count + 1

    # Test unpinning unregisters from PinMonitor
    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count

    # Test multiple pins/unpins
    memory_obj.pin()
    memory_obj.pin()  # Pin twice
    assert pin_monitor.get_monitored_count() == initial_count + 1

    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count + 1  # Still monitored

    memory_obj.unpin()
    assert pin_monitor.get_monitored_count() == initial_count  # Fully unregistered


def test_tensor_memory_obj_batch_pin_is_transactional(monkeypatch):
    allocator = TensorMemoryAllocator(torch.empty(16384, dtype=torch.uint8))
    memory_objs = allocator.batched_allocate(
        torch.Size([16]),
        torch.float32,
        3,
    )
    assert memory_objs is not None

    registered = []
    monitor = type(
        "_Monitor",
        (),
        {
            "on_pin_many": lambda _self, objs: registered.append(tuple(objs)),
            "on_unpin": lambda _self, _obj: None,
        },
    )()
    monkeypatch.setattr(PinMonitor, "GetOrCreate", lambda _config=None: monitor)
    initial_pinned = TensorMemoryObj.monitor.pinned_memory_objs_count

    TensorMemoryObj.pin_many(memory_objs)
    assert len(registered) == 1
    assert {id(obj) for obj in registered[0]} == {id(obj) for obj in memory_objs}
    assert all(obj.metadata.pin_count == 1 for obj in memory_objs)

    for obj in memory_objs:
        obj.unpin()

    monitor.on_pin_many = lambda _objs: (_ for _ in ()).throw(
        RuntimeError("registration failed")
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        TensorMemoryObj.pin_many(memory_objs)
    assert all(obj.metadata.pin_count == 0 for obj in memory_objs)
    assert TensorMemoryObj.monitor.pinned_memory_objs_count == initial_pinned

    for obj in memory_objs:
        obj.ref_count_down()


# =============================================================================
# LazyMemoryAllocator Tests
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LazyMemoryAllocator requires CUDA for memory pinning",
)
class TestLazyMemoryAllocator:
    """
    Test suite for LazyMemoryAllocator.

    These tests focus on the public interface defined by MemoryAllocatorInterface:
    - allocate(shapes, dtypes, fmt, allocator_type) -> Optional[MemoryObj]
    - batched_allocate(shapes, dtypes, batch_size, fmt, allocator_type)
        -> Optional[List[MemoryObj]]
    - free(memory_obj, allocator_type)
    - batched_free(memory_objs, allocator_type, update_stats)
    - close()
    - memcheck() -> bool
    """

    # Use sizes that are multiples of PIN_CHUNK_SIZE (16 MB)
    INIT_SIZE = 1 << 25  # 32 MB
    FINAL_SIZE = 1 << 27  # 128 MB

    @pytest.fixture
    def lazy_allocator_cls(self):
        """Lazily import LazyMemoryAllocator to avoid import errors
        on CPU-only builds.
        """
        # First Party
        from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator

        return LazyMemoryAllocator

    def test_allocate_basic(self, lazy_allocator_cls):
        """Test basic allocation returns a valid MemoryObj."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([512, 512])
        dtype = torch.float32
        memory_obj = allocator.allocate(shape, dtype)

        assert memory_obj is not None
        assert memory_obj.is_valid()
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.shape == shape
        assert memory_obj.tensor.dtype == dtype

        allocator.close()

    def test_allocate_with_format(self, lazy_allocator_cls):
        """Test allocation with explicit memory format."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 1024])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_T2D

        memory_obj = allocator.allocate(shape, dtype, fmt)

        assert memory_obj is not None
        assert memory_obj.is_valid()
        assert memory_obj.get_memory_format() == fmt

        allocator.close()

    def test_allocate_multiple_shapes_and_dtypes(self, lazy_allocator_cls):
        """Test allocation with multiple shapes and dtypes."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shapes = [torch.Size([100, 2, 512]), torch.Size([100, 2, 512])]
        dtypes = [torch.bfloat16, torch.bfloat16]

        memory_obj = allocator.allocate(shapes, dtypes)

        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.close()

    def test_allocate_returns_none_when_out_of_memory(self, lazy_allocator_cls):
        """Test that allocation returns None when memory is exhausted."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.INIT_SIZE,  # Same as init to prevent expansion
        )

        # Try to allocate more than available
        huge_shape = torch.Size([self.INIT_SIZE])
        memory_obj = allocator.allocate(huge_shape, torch.float32)

        assert memory_obj is None

        allocator.close()

    def test_free_basic(self, lazy_allocator_cls):
        """Test that free invalidates the MemoryObj."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([512, 512])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.free(memory_obj)
        assert not memory_obj.is_valid()
        assert memory_obj.tensor is None

        allocator.close()

    def test_free_idempotent(self, lazy_allocator_cls):
        """Test that freeing an already freed object does not crash."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([256, 256])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None

        allocator.free(memory_obj)
        # This should not crash
        allocator.free(memory_obj)

        assert allocator.memcheck()
        allocator.close()

    def test_batched_allocate_basic(self, lazy_allocator_cls):
        """Test batched allocation returns correct number of MemoryObjs."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        batch_size = 8

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is not None
        assert len(memory_objs) == batch_size
        for obj in memory_objs:
            assert obj is not None
            assert obj.is_valid()
            assert obj.tensor is not None
            assert obj.tensor.shape == shape
            assert obj.tensor.dtype == dtype

        allocator.close()

    def test_batched_allocate_with_format(self, lazy_allocator_cls):
        """Test batched allocation with explicit memory format."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_T2D
        batch_size = 4

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size, fmt)

        assert memory_objs is not None
        for obj in memory_objs:
            assert obj.get_memory_format() == fmt

        allocator.close()

    def test_batched_allocate_returns_none_when_out_of_memory(self, lazy_allocator_cls):
        """Test that batched allocation returns None when memory is exhausted."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.INIT_SIZE,
        )

        shape = torch.Size([1024 * 1024])  # 1M elements
        dtype = torch.float32  # 4 bytes each = 4MB per allocation
        batch_size = 100  # Would need 400MB, more than available

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)

        assert memory_objs is None

        allocator.close()

    def test_batched_free_basic(self, lazy_allocator_cls):
        """Test batched free invalidates all MemoryObjs."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([100, 2, 512])
        dtype = torch.bfloat16
        batch_size = 4

        memory_objs = allocator.batched_allocate(shape, dtype, batch_size)
        assert memory_objs is not None

        allocator.batched_free(memory_objs)

        for obj in memory_objs:
            assert not obj.is_valid()

        assert allocator.memcheck()
        allocator.close()

    def test_memcheck_returns_true_after_operations(self, lazy_allocator_cls):
        """Test that memcheck returns True after valid operations."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        # Initial state
        assert allocator.memcheck()

        # After allocation
        shape = torch.Size([512, 512])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert allocator.memcheck()

        # After free
        allocator.free(memory_obj)
        assert allocator.memcheck()

        # After batched operations
        objs = allocator.batched_allocate(shape, torch.float32, 4)
        assert allocator.memcheck()

        allocator.batched_free(objs)
        assert allocator.memcheck()

        allocator.close()

    def test_inplace_tensor_modification(self, lazy_allocator_cls):
        """Test that allocated tensor data can be modified in place."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([1024])
        memory_obj = allocator.allocate(shape, torch.float32)
        assert memory_obj is not None

        # Modify the tensor in place
        memory_obj.tensor.fill_(42.0)
        assert torch.all(memory_obj.tensor == 42.0)

        memory_obj.tensor[0] = 123.0
        assert memory_obj.tensor[0] == 123.0

        allocator.close()

    def test_lazy_expansion_allows_larger_allocations(self, lazy_allocator_cls):
        """
        Test that lazy expansion allows allocations beyond init_size.

        The background thread should expand the available memory over time,
        allowing allocations that exceed the initial size.
        """
        # Start with small init_size, larger final_size
        init_size = 1 << 25  # 32 MB
        final_size = 1 << 27  # 128 MB

        allocator = lazy_allocator_cls(
            init_size=init_size,
            final_size=final_size,
        )

        # Wait for background expansion to complete
        # This gives the lazy allocator time to expand memory
        time.sleep(0.5)

        # Try to allocate more than init_size (but less than final_size)
        # 64 MB > 32 MB init_size
        large_shape = torch.Size([16 * 1024 * 1024])  # 16M elements * 4 bytes = 64MB
        memory_obj = allocator.allocate(large_shape, torch.float32)

        assert memory_obj is not None
        assert memory_obj.is_valid()

        allocator.close()

    def test_allocate_various_dtypes(self, lazy_allocator_cls):
        """Test allocation with various data types."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        test_cases = [
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([1024, 1024]), torch.bfloat16),
            (torch.Size([2048, 2048]), torch.int8),
            (torch.Size([256, 256]), torch.half),
        ]

        memory_objs = []
        for shape, dtype in test_cases:
            obj = allocator.allocate(shape, dtype)
            assert obj is not None, f"Failed to allocate {shape} with {dtype}"
            assert obj.tensor.dtype == dtype
            assert obj.tensor.shape == shape
            memory_objs.append(obj)

        # Free all
        for obj in memory_objs:
            allocator.free(obj)

        assert allocator.memcheck()
        allocator.close()

    def test_allocation_and_free_interleaved(self, lazy_allocator_cls):
        """Test interleaved allocation and free operations."""
        allocator = lazy_allocator_cls(
            init_size=self.INIT_SIZE,
            final_size=self.FINAL_SIZE,
        )

        shape = torch.Size([256, 256])
        dtype = torch.float32

        obj1 = allocator.allocate(shape, dtype)
        obj2 = allocator.allocate(shape, dtype)

        allocator.free(obj1)

        obj3 = allocator.allocate(shape, dtype)

        allocator.free(obj2)
        allocator.free(obj3)

        assert allocator.memcheck()
        allocator.close()
