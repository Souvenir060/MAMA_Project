"""
Core module for the MAMA (Mature Agent Management Architecture) system.
Contains adaptive interaction protocols and system components.
"""

from .adaptive_interaction import (
    AdaptiveInteractionProtocol,
    InteractionRequest,
    InteractionResponse,
    InteractionMode,
    InteractionPriority,
    create_interaction_request
)

from .mcp_integration import (
    MCPContextManager,
    MCPClient,
    MCPMessage,
    MCPMessageType,
    get_mcp_manager
)

__all__ = [
    'AdaptiveInteractionProtocol',
    'InteractionRequest', 
    'InteractionResponse',
    'InteractionMode',
    'InteractionPriority',
    'create_interaction_request',
    'MCPContextManager',
    'MCPClient',
    'MCPMessage',
    'MCPMessageType',
    'get_mcp_manager'
]
