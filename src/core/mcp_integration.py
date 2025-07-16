# MAMA_exp/core/mcp_integration.py

"""
Model Context Protocol (MCP) Integration Module

This module implements the Model Context Protocol for seamless agent communication
and context sharing within the MAMA flight assistant system.
"""

import json
import logging
import asyncio
import websockets
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """MCP message types"""
    CONTEXT_REQUEST = "context_request"
    CONTEXT_RESPONSE = "context_response"
    AGENT_CALL = "agent_call"
    AGENT_RESPONSE = "agent_response"
    NOTIFICATION = "notification"
    ERROR = "error"

@dataclass
class MCPMessage:
    """MCP protocol message structure"""
    message_id: str
    message_type: MCPMessageType
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: str
    context_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "context_id": self.context_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MCPMessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            context_id=data.get("context_id")
        )

class MCPContextManager:
    """Manages shared context across agents using MCP"""
    
    def __init__(self):
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.context_subscribers: Dict[str, List[str]] = {}
        self.message_handlers: Dict[MCPMessageType, Callable] = {}
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MCPMessageType.CONTEXT_REQUEST] = self._handle_context_request
        self.message_handlers[MCPMessageType.AGENT_CALL] = self._handle_agent_call
        self.message_handlers[MCPMessageType.NOTIFICATION] = self._handle_notification
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start MCP WebSocket server"""
        try:
            server = await websockets.serve(
                self._handle_connection,
                host,
                port
            )
            logger.info(f"MCP server started on ws://{host}:{port}")
            return server
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        client_id = f"client_{id(websocket)}"
        self.active_connections[client_id] = websocket
        
        try:
            logger.info(f"New MCP connection: {client_id}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    mcp_message = MCPMessage.from_dict(data)
                    await self._process_message(mcp_message, client_id)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    await self._send_error(websocket, f"Message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"MCP connection closed: {client_id}")
        finally:
            self.active_connections.pop(client_id, None)
    
    async def _process_message(self, message: MCPMessage, client_id: str):
        """Process incoming MCP message"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message, client_id)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_context_request(self, message: MCPMessage, client_id: str):
        """Handle context request from agent"""
        context_id = message.payload.get("context_id")
        
        if context_id in self.contexts:
            response = MCPMessage(
                message_id=f"resp_{message.message_id}",
                message_type=MCPMessageType.CONTEXT_RESPONSE,
                sender_id="mcp_server",
                recipient_id=message.sender_id,
                payload={
                    "context_id": context_id,
                    "context_data": self.contexts[context_id]
                },
                timestamp=datetime.now().isoformat(),
                context_id=context_id
            )
        else:
            response = MCPMessage(
                message_id=f"resp_{message.message_id}",
                message_type=MCPMessageType.ERROR,
                sender_id="mcp_server",
                recipient_id=message.sender_id,
                payload={"error": f"Context {context_id} not found"},
                timestamp=datetime.now().isoformat()
            )
        
        await self._send_message_to_client(client_id, response)
    
    async def _handle_agent_call(self, message: MCPMessage, client_id: str):
        """Handle agent call through MCP"""
        target_agent = message.recipient_id
        
        # Route message to target agent if connected
        target_client = self._find_client_by_agent(target_agent)
        if target_client:
            await self._send_message_to_client(target_client, message)
        else:
            # Send error if target agent not available
            error_response = MCPMessage(
                message_id=f"error_{message.message_id}",
                message_type=MCPMessageType.ERROR,
                sender_id="mcp_server",
                recipient_id=message.sender_id,
                payload={"error": f"Agent {target_agent} not available"},
                timestamp=datetime.now().isoformat()
            )
            await self._send_message_to_client(client_id, error_response)
    
    async def _handle_notification(self, message: MCPMessage, client_id: str):
        """Handle notification broadcast"""
        # Broadcast notification to all connected clients
        for conn_id, websocket in self.active_connections.items():
            if conn_id != client_id:  # Don't send back to sender
                await self._send_message_to_client(conn_id, message)
    
    def _find_client_by_agent(self, agent_id: str) -> Optional[str]:
        """Find client connection by agent ID"""
        # This would need agent registration logic
        # For now, return None (agent routing not implemented)
        return None
    
    async def _send_message_to_client(self, client_id: str, message: MCPMessage):
        """Send message to specific client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send(json.dumps(message.to_dict()))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message through WebSocket"""
        error_msg = MCPMessage(
            message_id=f"error_{int(datetime.now().timestamp())}",
            message_type=MCPMessageType.ERROR,
            sender_id="mcp_server",
            recipient_id="client",
            payload={"error": error_message},
            timestamp=datetime.now().isoformat()
        )
        
        try:
            await websocket.send(json.dumps(error_msg.to_dict()))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    async def _send_error_to_client(self, client_id: str, error_message: str):
        """Send error message to specific client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            await self._send_error(websocket, error_message)
    
    def create_context(self, context_id: str, context_data: Dict[str, Any]) -> bool:
        """Create new shared context"""
        try:
            self.contexts[context_id] = {
                "data": context_data,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            logger.info(f"Created MCP context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create context {context_id}: {e}")
            return False
    
    def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing context"""
        try:
            if context_id in self.contexts:
                self.contexts[context_id]["data"].update(updates)
                self.contexts[context_id]["updated_at"] = datetime.now().isoformat()
                
                # Notify subscribers about context update
                asyncio.create_task(self._notify_context_subscribers(context_id, updates))
                
                logger.debug(f"Updated MCP context: {context_id}")
                return True
            else:
                logger.warning(f"Context {context_id} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to update context {context_id}: {e}")
            return False
    
    async def _notify_context_subscribers(self, context_id: str, updates: Dict[str, Any]):
        """Notify agents subscribed to context changes"""
        subscribers = self.context_subscribers.get(context_id, [])
        
        for subscriber in subscribers:
            notification = MCPMessage(
                message_id=f"notify_{int(datetime.now().timestamp())}",
                message_type=MCPMessageType.NOTIFICATION,
                sender_id="mcp_server",
                recipient_id=subscriber,
                payload={
                    "event": "context_updated",
                    "context_id": context_id,
                    "updates": updates
                },
                timestamp=datetime.now().isoformat(),
                context_id=context_id
            )
            
            client_id = self._find_client_by_agent(subscriber)
            if client_id:
                await self._send_message_to_client(client_id, notification)
    
    def subscribe_to_context(self, agent_id: str, context_id: str) -> bool:
        """Subscribe agent to context updates"""
        try:
            if context_id not in self.context_subscribers:
                self.context_subscribers[context_id] = []
            
            if agent_id not in self.context_subscribers[context_id]:
                self.context_subscribers[context_id].append(agent_id)
                logger.info(f"Agent {agent_id} subscribed to context {context_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe {agent_id} to context {context_id}: {e}")
            return False
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get context data"""
        return self.contexts.get(context_id, {}).get("data")

class MCPClient:
    """MCP client for agents to communicate through the protocol"""
    
    def __init__(self, agent_id: str, server_url: str = "ws://localhost:8765"):
        self.agent_id = agent_id
        self.server_url = server_url
        self.websocket = None
        self.message_handlers: Dict[MCPMessageType, Callable] = {}
        
    async def connect(self):
        """Connect to MCP server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Agent {self.agent_id} connected to MCP server")
            
            # Start message handling loop
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.websocket:
            await self.websocket.close()
            logger.info(f"Agent {self.agent_id} disconnected from MCP server")
    
    async def _handle_messages(self):
        """Handle incoming messages from MCP server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    mcp_message = MCPMessage.from_dict(data)
                    
                    handler = self.message_handlers.get(mcp_message.message_type)
                    if handler:
                        await handler(mcp_message)
                    else:
                        logger.debug(f"No handler for message type: {mcp_message.message_type}")
                        
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"MCP connection closed for agent {self.agent_id}")
    
    async def send_message(self, message: MCPMessage):
        """Send message through MCP"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(message.to_dict()))
            except Exception as e:
                logger.error(f"Failed to send MCP message: {e}")
                raise
    
    async def request_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Request context data from MCP server"""
        request = MCPMessage(
            message_id=f"req_{int(datetime.now().timestamp())}",
            message_type=MCPMessageType.CONTEXT_REQUEST,
            sender_id=self.agent_id,
            recipient_id="mcp_server",
            payload={"context_id": context_id},
            timestamp=datetime.now().isoformat(),
            context_id=context_id
        )
        
        await self.send_message(request)
        # In a real implementation, this would wait for the response
        # For now, return None
        return None

# Global MCP manager instance
mcp_manager = MCPContextManager()

def get_mcp_manager() -> MCPContextManager:
    """Get the global MCP manager instance"""
    return mcp_manager

async def start_mcp_server(host: str = "localhost", port: int = 8765):
    """Start the MCP server"""
    return await mcp_manager.start_server(host, port) 