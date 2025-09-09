import logging
import os
from typing import Dict, List, Optional, Iterator

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    WorkloadStatus,
    create_workload,
    delete_workload,
    get_workload,
    logs_workload,
    ContainerPort,
)

from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.utils.envs import sanitize_env
from gpustack.utils.hub import (
    get_max_model_len,
    get_pretrained_config,
)
from gpustack.utils.network import get_free_port
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class SGLangServer(InferenceServer):
    """
    Containerized SGLang inference server backend using gpustack-runtime.

    This backend runs SGLang in a Docker container managed by gpustack-runtime,
    providing better isolation, resource management, and deployment consistency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workload_name: Optional[str] = None

    def start(self):  # noqa: C901
        try:
            # Setup container mounts
            mounts = self._get_configured_mounts()

            # Setup environment variables
            envs = self._setup_environment()

            # Get resources configuration
            resources = self._get_configured_resources()

            # Get serving port
            serving_port = self._get_serving_port()

            # Build SGLang command arguments
            arguments = self._build_sglang_arguments(port=serving_port)

            # Get SGLang image name
            image_name = self._get_backend_image_name()

            if not image_name:
                raise ValueError("Can't find compatible SGLang image")

            # Create container configuration
            run_container = Container(
                image=image_name,
                name=self._model_instance.name,
                profile=ContainerProfileEnum.RUN,
                execution=ContainerExecution(
                    args=arguments,
                ),
                envs=[
                    ContainerEnv(name=name, value=value) for name, value in envs.items()
                ],
                mounts=mounts,
                resources=resources,
                ports=[
                    ContainerPort(
                        internal=serving_port,
                    ),
                ],
            )

            # Store workload name for management operations
            self._workload_name = self._model_instance.name

            workload_plan = WorkloadPlan(
                name=self._workload_name,
                host_network=True,
                containers=[run_container],
            )

            logger.info(f"Creating SGLang container workload: {self._workload_name}")
            logger.info(f"Container image name: {image_name} arguments: {arguments}")
            create_workload(workload_plan)

            logger.info(
                f"SGLang container workload {self._workload_name} created successfully"
            )

        except Exception as e:
            self._handle_error(e)

    def _setup_environment(self) -> Dict[str, str]:
        """
        Setup environment variables for the SGLang container server.
        """
        env = os.environ.copy()

        # Apply SGLang distributed environment setup
        self.set_sglang_distributed_env(env)

        # Apply GPUStack's inference environment setup
        env = self._get_configured_env()

        # Log environment variables
        env_view = None
        if logger.isEnabledFor(logging.DEBUG):
            env_view = sanitize_env(env)
        elif self._model.env:
            # If the model instance has its own environment variables,
            # display the mutated environment variables.
            env_view = self._model.env
            for k, v in self._model.env.items():
                env_view[k] = env.get(k, v)
        if env_view:
            logger.info(
                f"With environment variables(inconsistent input items mean unchangeable):{os.linesep}"
                f"{os.linesep.join(f'{k}={v}' for k, v in sorted(env_view.items()))}"
            )

        return env

    def _build_sglang_arguments(self, port: int) -> List[str]:
        """
        Build SGLang command arguments for container execution.
        """
        arguments = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_path,
        ]

        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(arguments)

        # Set host and port
        arguments.extend(
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]
        )

        # Add auto parallelism arguments if needed
        auto_parallelism_arguments = get_auto_parallelism_arguments(
            self._model.backend_parameters, self._model_instance
        )
        arguments.extend(auto_parallelism_arguments)

        # Add multi-node deployment parameters if needed
        multinode_arguments = self._get_multinode_arguments()
        arguments.extend(multinode_arguments)

        # Add user-defined backend parameters
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        return arguments

    def _get_multinode_arguments(self) -> List[str]:
        """
        Get multi-node deployment arguments for SGLang.
        """
        arguments = []

        # Check if this is a multi-node deployment
        if not (
            hasattr(self._model_instance, 'distributed_servers')
            and self._model_instance.distributed_servers
            and self._model_instance.distributed_servers.subordinate_workers
        ):
            return []

        subordinate_workers = (
            self._model_instance.distributed_servers.subordinate_workers
        )
        total_nodes = len(subordinate_workers) + 1  # +1 for the current node

        # Find the current node's rank
        current_worker_ip = self._worker.ip
        node_rank = 0  # Default to 0 (master node)
        is_main_worker = current_worker_ip == self._model_instance.worker_ip

        # Determine node rank based on worker IP
        if not is_main_worker:
            for idx, worker in enumerate(subordinate_workers):
                if worker.worker_ip == current_worker_ip:
                    node_rank = idx + 1  # Subordinate workers start from rank 1
                    break

        # Get or allocate distributed communication port
        dist_init_port = get_free_port(port_range=self._config.ray_worker_port_range)

        # Add multi-node parameters
        arguments.extend(
            [
                "--nnodes",
                str(total_nodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                f"{self._model_instance.worker_ip}:{dist_init_port}",
            ]
        )

        return arguments

    def _handle_error(self, error: Exception):
        """
        Handle errors during SGLang server startup.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run the vLLM container server: {error}{cause_text}"
        logger.exception(error_message)

        try:
            self._update_model_instance(
                self._model_instance.id,
                state=ModelInstanceStateEnum.ERROR,
                state_message=str(error),
            )
        except Exception as e:
            logger.error(f"Failed to update model instance: {e}")

    def set_sglang_distributed_env(self, env: Dict[str, str]):
        """
        Set up distributed environment variables for SGLang.
        """
        if is_distributed_sglang(self._model_instance):
            # Set up distributed training environment
            env["NCCL_DEBUG"] = "INFO"
            env["NCCL_SOCKET_IFNAME"] = "^lo,docker0"

            # Set master address and port for distributed training
            if (
                hasattr(self._model_instance, 'distributed_servers')
                and self._model_instance.distributed_servers
            ):
                subordinate_workers = (
                    self._model_instance.distributed_servers.subordinate_workers
                )
                if subordinate_workers:
                    master_worker = subordinate_workers[0]
                    env["MASTER_ADDR"] = master_worker.worker_ip or "localhost"
                    env["MASTER_PORT"] = str(
                        get_free_port(port_range=self._config.ray_worker_port_range)
                    )

    def _derive_max_model_len(self) -> Optional[int]:
        """
        Derive maximum model length from model configuration.
        """
        try:
            config = get_pretrained_config(self._model_path)
            return get_max_model_len(config)
        except Exception as e:
            logger.warning(f"Failed to derive max model length: {e}")
            return None

    def get_container_logs(
        self,
        tail: Optional[int] = 100,
        follow: bool = False,
        timestamps: bool = True,
        since: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Get container logs for the SGLang workload.
        """
        if not self._workload_name:
            logger.warning("No workload name available for log retrieval")
            return iter([])

        try:
            return logs_workload(
                name=self._workload_name,
                tail=tail,
                follow=follow,
                timestamps=timestamps,
                since=since,
            )
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return iter([])

    def get_container_status(self) -> Optional[WorkloadStatus]:
        """
        Get the status of the SGLang container workload.
        """
        if not self._workload_name:
            return None

        try:
            workload = get_workload(self._workload_name)
            return workload.status if workload else None
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return None

    def stop_container(self) -> bool:
        """
        Stop the SGLang container workload.
        """
        if not self._workload_name:
            logger.warning("No workload name available for stopping")
            return False

        try:
            delete_workload(self._workload_name)
            logger.info(f"SGLang container workload {self._workload_name} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def restart_container(self) -> bool:
        """
        Restart the SGLang container workload.
        """
        try:
            # Stop the current container
            if not self.stop_container():
                return False

            # Start a new container
            self.start()
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False


def get_auto_parallelism_arguments(
    backend_parameters: List[str], model_instance: ModelInstance
) -> List[str]:
    """
    Get auto parallelism arguments for SGLang based on GPU configuration.
    """
    arguments = []

    # Check if tensor parallelism is already specified
    if model_instance.gpu_indexes and "--tp-size" not in backend_parameters:
        gpu_count = len(model_instance.gpu_indexes)
        if gpu_count > 1:
            arguments.extend(["--tp-size", str(gpu_count)])

    return arguments


def is_distributed_sglang(model_instance: ModelInstance) -> bool:
    """
    Check if the model instance requires to be distributed SGLang setup.
    """
    # Check for multi-node deployment
    if (
        hasattr(model_instance, 'distributed_servers')
        and model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    ):
        return True

    # Check for multi-GPU on single node
    if not model_instance.gpu_indexes:
        return False

    # Consider it distributed if using multiple GPUs
    return len(model_instance.gpu_indexes) > 1
