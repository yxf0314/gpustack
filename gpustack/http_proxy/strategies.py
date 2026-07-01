from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
from typing import Dict, List
import itertools

from gpustack.schemas.models import ModelInstance

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(ABC):

    @abstractmethod
    async def select_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    def __init__(self):
        self._iterators: Dict[int, itertools.cycle] = {}
        self._instance_lists: Dict[int, List[ModelInstance]] = {}

    async def select_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        if len(instances) == 0:
            raise Exception("No instances available")
        model_id = instances[0].model_id
        if (
            model_id not in self._iterators
            or self._instance_lists[model_id] != instances
        ):
            logger.debug(f"Creating new iterator for model {model_id}")
            self._iterators[model_id] = itertools.cycle(instances)
            self._instance_lists[model_id] = instances

        return next(self._iterators[model_id])


class WorkerRoundRobinStrategy:
    """Round-robins over the worker ids serving a single model instance.

    Used for vLLM hybrid-LB / external-LB, where the leader and every subordinate
    expose their own API, so one logical instance spans several serving
    endpoints. Keyed by instance id; the cycle resets when the worker set changes
    (scale up/down or a worker dropping out). This is orthogonal to
    RoundRobinStrategy, which balances across the replicas (instances) of a model.
    """

    # Deleted instances are never removed, so bound the map with an LRU.
    _max_instances = 1024

    def __init__(self):
        self._iterators: OrderedDict[int, itertools.cycle] = OrderedDict()
        self._worker_id_lists: Dict[int, List[int]] = {}

    def select_worker_id(self, instance_id: int, worker_ids: List[int]) -> int:
        if len(worker_ids) == 0:
            raise Exception("No worker ids available")
        if (
            instance_id not in self._iterators
            or self._worker_id_lists[instance_id] != worker_ids
        ):
            self._iterators[instance_id] = itertools.cycle(worker_ids)
            self._worker_id_lists[instance_id] = list(worker_ids)
            while len(self._iterators) > self._max_instances:
                evicted_id, _ = self._iterators.popitem(last=False)
                self._worker_id_lists.pop(evicted_id, None)
        self._iterators.move_to_end(instance_id)
        return next(self._iterators[instance_id])
