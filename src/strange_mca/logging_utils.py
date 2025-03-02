"""Logging utilities for the multiagent system."""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class DetailedLoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs detailed information about LLM calls and node traversal."""
    
    def __init__(self, verbose: bool = True, log_level: str = "warn"):
        """Initialize the callback handler.
        
        Args:
            verbose: Whether to enable verbose output.
            log_level: The level of logging detail using standard Python logging levels: "warn", "info", or "debug".
                       Default is "warn" which shows only warnings and errors.
        """
        super().__init__()
        self.verbose = verbose
        self.log_level = log_level
        self.logger = logging.getLogger("strange_mca")
        
        # Map string log levels to Python logging levels
        self.log_level_map = {
            "warn": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG
        }
        
        # Set the logger level based on the provided log_level
        if log_level in self.log_level_map:
            self.logger.setLevel(self.log_level_map[log_level])
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log when a chain starts."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        self.logger.debug(f"[CHAIN_START:{node_name}] Node {node_name} started")
        
        # Log the task if available (only in detailed mode)
        if "task" in inputs:
            task_preview = inputs['task'][:50] + "..." if len(inputs['task']) > 50 else inputs['task']
            self.logger.debug(f"[CHAIN_START:{node_name}] Task for {node_name}: {task_preview}")
    
    def on_chain_end(
        self, outputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log when a chain ends."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Log the response if available (only in detailed mode)
        if "responses" in outputs and isinstance(outputs["responses"], dict):
            for agent_name, response in outputs["responses"].items():
                response_preview = response[:50] + "..." if len(response) > 50 else response
                self.logger.debug(f"[CHAIN_END:{node_name}] Response from {agent_name}: {response_preview}")
        
        self.logger.debug(f"[CHAIN_END:{node_name}] Node {node_name} completed")
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log when an LLM starts."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        

        self.logger.info(f"[LLM_START:{node_name}] LLM call started (run_id: {run_id_short})")
        
        if prompts:
            prompt_preview = prompts[0][:30] + "..." if len(prompts[0]) > 30 else prompts[0]
            self.logger.info(f"[LLM_START:{node_name}] Prompt: {prompt_preview}")
    
    def on_llm_end(
        self, response: Any, **kwargs: Any
    ) -> None:
        """Log when an LLM ends."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Try to extract the content from the response
        content = ""
        if hasattr(response, "generations"):
            generations = response.generations
            if generations and generations[0]:
                content = generations[0][0].text
        
        # Only show a short preview
        content_preview = content[:30] + "..." if len(content) > 30 else content
        
        # Log at debug level (only shown in detailed mode)
        self.logger.info(f"[LLM_END:{node_name}] LLM response (run_id: {run_id_short}): {content_preview}")
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> None:
        """Log when a chat model starts."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[CHAT_START:{node_name}] Chat model call started (run_id: {run_id_short})")
        
        # Log message count at debug level
        if messages:
            message_count = sum(len(msg_list) for msg_list in messages)
            self.logger.debug(f"[CHAT_START:{node_name}] Messages: {message_count} total")
            
    def on_chat_model_end(
        self, response: Any, **kwargs: Any
    ) -> None:
        """Log when a chat model ends."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Try to extract the content from the response
        content = ""
        if hasattr(response, "generations"):
            generations = response.generations
            if generations and generations[0]:
                content = generations[0][0].text
        
        # Only show a short preview
        content_preview = content[:30] + "..." if len(content) > 30 else content
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[CHAT_END:{node_name}] Chat model response (run_id: {run_id_short}): {content_preview}")
        
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Log when a tool starts."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Extract tool name if available
        tool_name = serialized.get("name", "unknown_tool")
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[TOOL_START:{node_name}] Tool '{tool_name}' started (run_id: {run_id_short})")
        
        # Log input preview at debug level
        input_preview = input_str[:30] + "..." if len(input_str) > 30 else input_str
        self.logger.debug(f"[TOOL_START:{node_name}] Input: {input_preview}")
        
    def on_tool_end(
        self, output: str, **kwargs: Any
    ) -> None:
        """Log when a tool ends."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[TOOL_END:{node_name}] Tool completed (run_id: {run_id_short})")
        
        # Log output preview at debug level
        output_preview = output[:30] + "..." if len(output) > 30 else output
        self.logger.debug(f"[TOOL_END:{node_name}] Output: {output_preview}")
        
    def on_agent_action(
        self, action: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log when an agent takes an action."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Extract action details
        action_name = action.get("tool", "unknown_action")
        action_input = action.get("tool_input", "")
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[AGENT_ACTION:{node_name}] Agent taking action '{action_name}' (run_id: {run_id_short})")
        
        # Log input preview at debug level
        if action_input:
            input_preview = str(action_input)[:30] + "..." if len(str(action_input)) > 30 else str(action_input)
            self.logger.debug(f"[AGENT_ACTION:{node_name}] Action input: {input_preview}")
            
    def on_agent_finish(
        self, finish: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log when an agent finishes."""
        if not self.verbose:
            return
            
        # Extract node name if available
        node_name = kwargs.get("tags", ["unknown_node"])[0] if kwargs.get("tags") else "unknown_node"
        
        # Extract finish details
        return_value = finish.get("return_values", {})
        
        # Convert run_id to string to avoid UUID subscriptability issues
        run_id = str(kwargs.get("run_id", "unknown"))
        run_id_short = run_id[:8] if len(run_id) > 8 else run_id
        
        # Log at debug level (only shown in detailed mode)
        self.logger.debug(f"[AGENT_FINISH:{node_name}] Agent finished (run_id: {run_id_short})")
        
        # Log output preview at debug level
        if "output" in return_value:
            output = return_value["output"]
            output_preview = str(output)[:30] + "..." if len(str(output)) > 30 else str(output)
            self.logger.debug(f"[AGENT_FINISH:{node_name}] Output: {output_preview}")


def setup_detailed_logging(level: int = None, log_level: str = "warn", only_local_logs: bool = False) -> None:
    """Set up detailed logging for the application.
    
    Args:
        level: The logging level to use (if specified, overrides log_level).
        log_level: The level of logging detail using standard Python logging levels: "warn", "info", or "debug".
                   Default is "warn" which shows only warnings and errors.
        only_local_logs: If True, only show logs from the strange_mca logger and suppress logs from other loggers.
    """
    # Map string log levels to Python logging levels
    log_level_map = {
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG
    }
    
    # Determine the actual logging level to use
    actual_level = level if level is not None else log_level_map.get(log_level, logging.WARNING)
    
    # Configure the root logger
    logging.basicConfig(
        level=actual_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set the level for all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        if only_local_logs and logger_name != "strange_mca" and not logger_name.startswith("strange_mca."):
            # If only_local_logs is True, set other loggers to a higher level to suppress their output
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        else:
            logging.getLogger(logger_name).setLevel(actual_level)
    
    # Create a logger for the strange_mca package
    logger = logging.getLogger("strange_mca")
    logger.setLevel(actual_level)
    
    # Ensure we don't duplicate log messages
    logger.propagate = False
    
    # Add a handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return log_level 