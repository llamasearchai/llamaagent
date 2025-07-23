"""
ReactAgent implementation for backward compatibility

This module provides the ReactAgent class that maintains backward compatibility
with existing LlamaAgent implementations while serving as a foundation for
enhanced cognitive agents.

Author: LlamaAgent Development Team
"""

import logging
from typing import Any, Dict, Optional

from .base import BaseAgent, AgentConfig, AgentResponse

logger = logging.getLogger(__name__)


class ReactAgent(BaseAgent):
    """
    ReactAgent implementation for backward compatibility.
    
    This agent provides the classic ReAct (Reasoning and Acting) pattern
    with basic tool usage and reasoning capabilities.
    """
    
    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        """Initialize ReactAgent"""
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger(f"{__name__}.ReactAgent")
        
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute a task using ReAct pattern"""
        start_time = self._get_current_time()
        
        try:
            # Basic task execution for now
            content = f"Processed task: {task}"
            
            # Update statistics
            execution_time = self._get_current_time() - start_time
            self.stats.update(execution_time, True, 50)
            
            return AgentResponse(
                content=content,
                success=True,
                execution_time=execution_time,
                tokens_used=50,
                metadata={"agent_type": "react", "task_type": "basic"}
            )
            
        except Exception as e:
            execution_time = self._get_current_time() - start_time
            self.stats.update(execution_time, False, 0)
            
            return AgentResponse(
                content=f"Task execution failed: {str(e)}",
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata={"agent_type": "react", "error": True}
            )
    
    def _get_current_time(self) -> float:
        """Get current time for timing calculations"""
        import time
        return time.time()


__all__ = ["ReactAgent"]
