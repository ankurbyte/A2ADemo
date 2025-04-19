from typing import AsyncIterable, List
from data_analysis_agent import DataAnalysisAgent
from common.server.task_manager import InMemoryTaskManager
from common.server import utils
from common.types import (
    Artifact,
    JSONRPCResponse,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
    Message,
    Part
)
import logging

logger = logging.getLogger(__name__)

class AgentTaskManager(InMemoryTaskManager):
    """Agent Task Manager for data analysis."""

    def __init__(self, agent: DataAnalysisAgent):
        super().__init__()
        self.agent = agent
        self.tasks = {}  # Add this line to store tasks

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def upsert_task(self, task: Task) -> None:
        """Update or insert a task."""
        self.tasks[task.id] = task
        
    async def _stream_generator(
        self, request: SendTaskRequest
    ) -> AsyncIterable[SendTaskResponse]:
        raise NotImplementedError("Not implemented")

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Handle subscription requests for task updates."""
        raise NotImplementedError("Streaming subscriptions are not supported")

    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        if not utils.are_modalities_compatible(
            request.params.acceptedOutputModes,
            DataAnalysisAgent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                request.params.acceptedOutputModes,
                DataAnalysisAgent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

        task_send_params: TaskSendParams = request.params
        await self.upsert_task(task_send_params)

        return await self._invoke(request)

    async def _invoke(self, request: SendTaskRequest) -> SendTaskResponse:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        
        try:
            result = self.agent.invoke(query)
            
            # Create a new Task with the response
            task = Task(
                id=task_send_params.id,
                message=Message(
                    role="agent",  # Changed from 'assistant' to 'agent'
                    parts=[TextPart(text=result.content, type="text")],
                ),
                status=TaskStatus(state=TaskState.COMPLETED),
                artifacts=[Artifact(parts=[TextPart(text=result.content, type="text")])]
            )
            
            await self.upsert_task(task)
            logger.info(f"Sending response to UI: {result.content[:100]}...")
            
            return SendTaskResponse(id=request.id, result=task)
        except Exception as e:
            logger.error("Error invoking agent: %s", e)
            error_task = Task(
                id=task_send_params.id,
                message=Message(
                    role="agent",  # Changed from 'assistant' to 'agent'
                    parts=[TextPart(text=f"Error: {str(e)}", type="text")],
                ),
                status=TaskStatus(state=TaskState.ERROR),
                artifacts=[]
            )
            await self.upsert_task(error_task)
            raise ValueError(f"Error invoking agent: {e}") from e

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            raise ValueError("Only text parts are supported")
        return part.text

    async def _update_store(
        self,
        task_id: str,
        status: TaskStatus,
        artifacts: List[Artifact] | None = None,
    ) -> Task:
        """Update task status and artifacts in store."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        task.status = status
        if artifacts:
            task.artifacts = artifacts
        
        await self.upsert_task(task)
        return task