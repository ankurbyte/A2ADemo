import sys
import os

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from common.server.task_manager import InMemoryTaskManager  # task_manager is in server subdirectory
from common.types import (
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskSendParams,
    Task,
    TaskStatus,
    TaskState,
    Message,
    Artifact,
    TextPart,
    JSONRPCResponse,
    InternalError,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent
)
from common.server.utils import are_modalities_compatible, new_incompatible_types_error  # Changed from importing utils
from typing import AsyncIterable, Union
import logging
from .CustomerDetails_agent import HiAgentCustomerAgent  # Changed to relative import

logger = logging.getLogger(__name__)

class AgentTaskManager(InMemoryTaskManager):
    """Task Manager for HiAgent customer details agent."""

    def __init__(self, agent: HiAgentCustomerAgent):
        super().__init__()
        self.agent = agent
        self._tasks = {}  # Add task store dictionary

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    async def upsert_task(self, task: Task | TaskSendParams) -> None:
        """Store or update a task."""
        if isinstance(task, TaskSendParams):
            # Convert TaskSendParams to Task if needed
            task = Task(
                id=task.id,
                sessionId=task.sessionId,
                status=TaskStatus(state=TaskState.WORKING),
                message=task.message
            )
        self._tasks[task.id] = task

    async def _stream_generator(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)

        try:
            async for item in self.agent.stream(query, task_send_params.sessionId):
                is_task_complete = item["is_task_complete"]
                require_user_input = item.get("require_user_input", False)
                
                parts = [{"type": "text", "text": item["content"]}]
                end_stream = False
                
                if not is_task_complete and not require_user_input:
                    task_state = TaskState.WORKING
                elif require_user_input:
                    task_state = TaskState.INPUT_REQUIRED
                    end_stream = True
                else:
                    task_state = TaskState.COMPLETED
                    end_stream = True
                
                message = Message(role="agent", parts=parts)
                task_status = TaskStatus(state=task_state, message=message)
                
                # Update task store and send notification
                artifacts = [Artifact(parts=parts)] if is_task_complete else None
                latest_task = await self._update_store(
                    task_send_params.id,
                    task_status,
                    artifacts
                )

                # Yield task status update
                yield SendTaskStreamingResponse(
                    id=request.id,
                    result=TaskStatusUpdateEvent(
                        id=task_send_params.id,
                        status=task_status,
                        final=end_stream
                    )
                )

                # Yield artifacts if present
                if artifacts:
                    for artifact in artifacts:
                        yield SendTaskStreamingResponse(
                            id=request.id,
                            result=TaskArtifactUpdateEvent(
                                id=task_send_params.id,
                                artifact=artifact
                            )
                        )

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield SendTaskStreamingResponse(
                id=request.id,
                error=InternalError(message=str(e))
            )

    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> JSONRPCResponse | None:
        task_send_params: TaskSendParams = request.params
        if not are_modalities_compatible(  # Changed from utils.are_modalities_compatible
            task_send_params.acceptedOutputModes,
            HiAgentCustomerAgent.SUPPORTED_CONTENT_TYPES
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                task_send_params.acceptedOutputModes,
                HiAgentCustomerAgent.SUPPORTED_CONTENT_TYPES,
            )
            return new_incompatible_types_error(request.id)  # Changed from utils.new_incompatible_types_error
        return None

    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse:
        """Handle non-streaming task requests."""
        error = self._validate_request(request)
        if error:
            return error

        await self.upsert_task(request.params)
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)

        try:
            result = self.agent.invoke(query, task_send_params.sessionId)
            parts = [{"type": "text", "text": result["content"]}]
            
            task_state = TaskState.COMPLETED
            if result.get("require_user_input", False):
                task_state = TaskState.INPUT_REQUIRED

            task = await self._update_store(
                task_send_params.id,
                TaskStatus(
                    state=task_state,
                    message=Message(role="agent", parts=parts)
                ),
                [Artifact(parts=parts)]
            )
            return SendTaskResponse(id=request.id, result=task)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise ValueError(f"Error invoking agent: {e}")

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handle streaming task requests."""
        error = self._validate_request(request)
        if error:
            return error

        await self.upsert_task(request.params)
        return self._stream_generator(request)

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            raise ValueError("Only text parts are supported")
        return part.text

    async def _update_store(
        self,
        task_id: str,
        status: TaskStatus,
        artifacts: list[Artifact] | None = None
    ) -> Task:
        """Updates the task store with new status and artifacts."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        task.status = status
        if artifacts:
            task.artifacts = artifacts
        
        await self.upsert_task(task)
        return task