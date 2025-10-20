# =====================================
# src/orchestrator/task_manager.py
# =====================================
"""
Task management for the orchestrator
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum


class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskContext:
    """Context for task execution"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


class TaskManager:
    """Manages task lifecycle and execution"""

    def __init__(self):
        self.tasks: Dict[str, Any] = {}
        self.task_queue: List[str] = []

    async def create_task(
            self,
            request: str,
            context: Optional[TaskContext] = None
    ) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "id": task_id,
            "request": request,
            "context": context or TaskContext(),
            "status": TaskStatus.PENDING,
            "created_at": datetime.now(),
            "subtasks": []
        }
        self.task_queue.append(task_id)
        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    async def update_task_status(
            self,
            task_id: str,
            status: TaskStatus,
            result: Optional[Any] = None
    ) -> None:
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            if result:
                self.tasks[task_id]["result"] = result
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self.tasks[task_id]["completed_at"] = datetime.now()


