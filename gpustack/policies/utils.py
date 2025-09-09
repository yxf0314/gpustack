import logging
import os
from typing import List, Optional
from gpustack.policies.base import (
    Allocatable,
    Allocated,
)
from gpustack.schemas.models import (
    ModelInstance,
)
from gpustack.schemas.workers import Worker, GPUDevicesInfo, GPUDeviceInfo
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WorkerGPUInfo(BaseModel):
    """
    Data structure to represent a GPU device with its associated worker information.
    """

    worker_id: int
    worker_name: str
    gpu_device: GPUDeviceInfo
    allocatable_vram: int  # in bytes


async def get_worker_allocatable_resource(  # noqa: C901
    engine: AsyncEngine,
    worker: Worker,
) -> Allocatable:
    """
    Get the worker with the latest allocatable resources.
    """

    def update_allocated_vram(allocated, resource_claim):
        for gpu_index, vram in resource_claim.vram.items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    is_unified_memory = worker.status.memory.is_unified_memory
    model_instances = await get_worker_model_instances(engine, worker)
    allocated = Allocated(ram=0, vram={})

    for model_instance in model_instances:
        if model_instance.worker_id != worker.id:
            continue
        allocated.ram += model_instance.computed_resource_claim.ram or 0
        if model_instance.gpu_indexes:
            update_allocated_vram(allocated, model_instance.computed_resource_claim)

        if (
            model_instance.distributed_servers
            and model_instance.distributed_servers.subordinate_workers
        ):
            for (
                subordinate_worker
            ) in model_instance.distributed_servers.subordinate_workers:
                if subordinate_worker.worker_id != worker.id:
                    continue

                if subordinate_worker.computed_resource_claim:
                    # rpc server only consider the vram
                    update_allocated_vram(
                        allocated, subordinate_worker.computed_resource_claim
                    )

    allocatable = Allocatable(ram=0, vram={})
    if worker.status.gpu_devices:
        for _, gpu in enumerate(worker.status.gpu_devices):
            gpu_index = gpu.index

            if gpu.memory is None or gpu.memory.total is None:
                continue
            allocatable_vram = max(
                (
                    gpu.memory.total
                    - allocated.vram.get(gpu_index, 0)
                    - worker.system_reserved.vram
                ),
                0,
            )
            allocatable.vram[gpu_index] = allocatable_vram

    allocatable.ram = max(
        (worker.status.memory.total - allocated.ram - worker.system_reserved.ram), 0
    )

    if is_unified_memory:
        allocatable.ram = max(
            allocatable.ram
            - worker.system_reserved.vram
            - sum(allocated.vram.values()),
            0,
        )

        # For UMA, we need to set the gpu memory to the minimum of
        # the calculated with max allow gpu memory and the allocatable memory.
        if allocatable.vram:
            allocatable.vram[0] = min(allocatable.ram, allocatable.vram[0])

    logger.debug(
        f"Worker {worker.name} reserved memory: {worker.system_reserved.ram}, "
        f"reserved gpu memory: {worker.system_reserved.vram}, "
        f"allocatable memory: {allocatable.ram}, "
        f"allocatable gpu memory: {allocatable.vram}"
    )
    return allocatable


def group_gpu_devices_by_memory(gpu_devices: GPUDevicesInfo) -> List[GPUDevicesInfo]:
    """
    Group GPU devices by allocatable memory size with the constraint that the minimum
    allocatable GPU memory in each group should not be less than 0.9 times the
    allocatable memory of other GPUs in the same group.

    Args:
        gpu_devices: List of GPU device information

    Returns:
        List of GPU device groups, where each group is a list of GPU devices

    Example:
        If we have GPUs with allocatable memory [8GB, 8.5GB, 16GB, 16.5GB, 32GB],
        they might be grouped as:
        - Group 1: [8GB, 8.5GB] (8GB >= 8.5GB * 0.9 = 7.65GB)
        - Group 2: [16GB, 16.5GB] (16GB >= 16.5GB * 0.9 = 14.85GB)
        - Group 3: [32GB]
    """
    if not gpu_devices:
        return []

    def get_allocatable_memory(gpu: GPUDeviceInfo) -> Optional[int]:
        """Calculate allocatable memory (total - allocated)"""
        if not gpu.memory or gpu.memory.total is None:
            return None
        allocated = gpu.memory.allocated or 0
        return gpu.memory.total - allocated

    # Filter out GPUs without valid memory information
    valid_gpus = []
    for gpu in gpu_devices:
        allocatable_memory = get_allocatable_memory(gpu)
        if allocatable_memory is not None and allocatable_memory > 0:
            valid_gpus.append(gpu)

    if not valid_gpus:
        return []

    # Sort GPUs by allocatable memory size (ascending order)
    sorted_gpus = sorted(valid_gpus, key=lambda gpu: get_allocatable_memory(gpu))

    groups = []
    current_group = []

    for gpu in sorted_gpus:
        if not current_group:
            # Start a new group
            current_group = [gpu]
        else:
            # Check if this GPU can be added to the current group
            min_allocatable_memory = get_allocatable_memory(
                current_group[0]
            )  # First GPU has minimum allocatable memory
            current_gpu_allocatable_memory = get_allocatable_memory(gpu)

            # Check if min_allocatable_memory >= current_gpu_allocatable_memory * 0.9
            # This ensures the minimum allocatable memory is not less than 0.9 times any other allocatable memory in the group
            if min_allocatable_memory >= current_gpu_allocatable_memory * 0.9:
                current_group.append(gpu)
            else:
                # Cannot add to current group, start a new group
                groups.append(current_group)
                current_group = [gpu]

    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups


async def get_worker_model_instances(
    engine: AsyncEngine, worker: Worker
) -> List[ModelInstance]:
    async with AsyncSession(engine) as session:
        model_instances = await ModelInstance.all_by_field(
            session, "worker_id", worker.id
        )
        return model_instances


class ListMessageBuilder:
    def __init__(self, messages: Optional[str | List[str]]):
        if not messages:
            self._messages = []
        self._messages = messages if isinstance(messages, list) else [messages]

    def append(self, message: str):
        self._messages.append(message)

    def extend(self, message: List[str]):
        self._messages.extend(message)

    def __str__(self) -> str:
        return "\n".join([f"- {line}" for line in self._messages])


def get_model_num_attention_heads(pretrained_config) -> Optional[int]:
    """
    Get the number of attention heads in the model.
    Priority: llm_config > text_config > root-level setting > thinker_config.text_config
    """

    num_attention_heads = None
    try:
        # Helper to get num_attention_heads from config
        def get_heads_from(cfg, key="num_attention_heads"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
            return None

        thinker_config = getattr(pretrained_config, "thinker_config", None)
        thinker_text_config = (
            getattr(thinker_config, "text_config", None) if thinker_config else None
        )

        configs_by_priority = [
            getattr(pretrained_config, "llm_config", None),
            getattr(pretrained_config, "text_config", None),
            pretrained_config,
            thinker_text_config,
        ]

        for config in configs_by_priority:
            if not config:
                continue
            heads = get_heads_from(config)
            if heads is not None:
                num_attention_heads = heads
                break

    except Exception as e:
        logger.warning(f"Cannot get num_attention_heads: {e}")

    return num_attention_heads


def get_local_model_weight_size(local_path: str) -> int:
    """
    Get the local model weight size in bytes. Estimate by the total size of files in the top-level (depth 1) of the directory.
    """
    total_size = 0

    try:
        with os.scandir(local_path) as entries:
            for entry in entries:
                if entry.is_file():
                    total_size += entry.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{local_path}' does not exist.")
    except NotADirectoryError:
        raise NotADirectoryError(
            f"The specified path '{local_path}' is not a directory."
        )
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing '{local_path}'.")

    return total_size


async def group_worker_gpu_by_memory(
    engine: AsyncEngine, workers: List[Worker]
) -> List[List[WorkerGPUInfo]]:
    """
    Group GPU devices from multiple workers by allocatable memory size with the constraint
    that the minimum allocatable GPU memory in each group should not be less than 0.9 times
    the allocatable memory of other GPUs in the same group.

    Args:
        engine: Database engine for calculating allocatable resources
        workers: List of workers containing GPU devices

    Returns:
        List of GPU device groups, where each group is a list of WorkerGPUInfo objects
        containing worker information and GPU device details

    Example:
        If we have GPUs from different workers with allocatable memory [8GB, 8.5GB, 16GB, 16.5GB, 32GB],
        they might be grouped as:
        - Group 1: [WorkerGPUInfo(worker1, gpu1, 8GB), WorkerGPUInfo(worker2, gpu2, 8.5GB)]
        - Group 2: [WorkerGPUInfo(worker1, gpu3, 16GB), WorkerGPUInfo(worker3, gpu1, 16.5GB)]
        - Group 3: [WorkerGPUInfo(worker2, gpu4, 32GB)]
    """
    if not workers:
        return []

    # Collect all GPU devices with their worker information and allocatable VRAM
    worker_gpu_infos = []

    for worker in workers:
        if not worker.status or not worker.status.gpu_devices:
            continue

        # Get allocatable resources for this worker
        allocatable = await get_worker_allocatable_resource(engine, worker)

        for gpu_device in worker.status.gpu_devices:
            if gpu_device.index is None:
                continue

            # Get allocatable VRAM for this specific GPU
            gpu_index = gpu_device.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)

            # Only include GPUs with positive allocatable VRAM
            if allocatable_vram > 0:
                worker_gpu_info = WorkerGPUInfo(
                    worker_id=worker.id,
                    worker_name=worker.name,
                    gpu_device=gpu_device,
                    allocatable_vram=allocatable_vram,
                )
                worker_gpu_infos.append(worker_gpu_info)

    if not worker_gpu_infos:
        return []

    # Sort GPUs by allocatable VRAM size (ascending order)
    sorted_worker_gpu_infos = sorted(
        worker_gpu_infos, key=lambda info: info.allocatable_vram
    )

    groups = []
    current_group = []

    for worker_gpu_info in sorted_worker_gpu_infos:
        if not current_group:
            # Start a new group
            current_group = [worker_gpu_info]
        else:
            # Check if this GPU can be added to the current group
            min_allocatable_vram = current_group[
                0
            ].allocatable_vram  # First GPU has minimum allocatable VRAM
            current_allocatable_vram = worker_gpu_info.allocatable_vram

            # Check if min_allocatable_vram >= current_allocatable_vram * 0.9
            # This ensures the minimum allocatable VRAM is not less than 0.9 times any other allocatable VRAM in the group
            if min_allocatable_vram >= current_allocatable_vram * 0.9:
                current_group.append(worker_gpu_info)
            else:
                # Cannot add to current group, start a new group
                groups.append(current_group)
                current_group = [worker_gpu_info]

    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups
