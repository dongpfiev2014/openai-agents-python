"""Memory and conversation management for OpenAI Agents SDK Demo"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import logging


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    timestamp: str
    role: str
    content: str
    metadata: Dict[str, Any] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
            
        return result


class Memory(ABC):
    """Abstract base class for memory implementations"""
    
    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str, **kwargs) -> None:
        """Add a message to memory"""
        pass
    
    @abstractmethod
    def get_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        pass
    
    @abstractmethod
    def clear(self, conversation_id: str = None) -> None:
        """Clear memory"""
        pass
    
    @abstractmethod
    def save(self, file_path: str) -> None:
        """Save memory to file"""
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load memory from file"""
        pass


class ConversationMemory(Memory):
    """In-memory conversation storage with persistence support"""
    
    def __init__(self, max_history: int = 50, auto_save: bool = False, save_path: str = None):
        self.max_history = max_history
        self.auto_save = auto_save
        self.save_path = save_path
        self.conversations: Dict[str, List[MemoryEntry]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing data if save_path exists
        if self.save_path and os.path.exists(self.save_path):
            try:
                self.load(self.save_path)
            except Exception as e:
                self.logger.warning(f"Failed to load memory from {self.save_path}: {e}")
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   tool_calls: Optional[List[Dict]] = None,
                   tool_call_id: Optional[str] = None,
                   name: Optional[str] = None,
                   metadata: Dict[str, Any] = None) -> None:
        """Add a message to the conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id].append(entry)
        
        # Trim history if it exceeds max_history
        if len(self.conversations[conversation_id]) > self.max_history:
            # Keep system messages and trim user/assistant messages
            system_messages = [msg for msg in self.conversations[conversation_id] if msg.role == "system"]
            other_messages = [msg for msg in self.conversations[conversation_id] if msg.role != "system"]
            
            # Keep the most recent messages
            trimmed_messages = other_messages[-(self.max_history - len(system_messages)):]
            self.conversations[conversation_id] = system_messages + trimmed_messages
        
        # Auto-save if enabled
        if self.auto_save and self.save_path:
            try:
                self.save(self.save_path)
            except Exception as e:
                self.logger.error(f"Failed to auto-save memory: {e}")
    
    def get_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries for API calls"""
        if conversation_id is None:
            # Return all conversations combined
            all_messages = []
            for conv_messages in self.conversations.values():
                all_messages.extend([msg.to_dict() for msg in conv_messages])
            return all_messages
        
        if conversation_id not in self.conversations:
            return []
        
        return [msg.to_dict() for msg in self.conversations[conversation_id]]
    
    def get_raw_history(self, conversation_id: str = None) -> List[MemoryEntry]:
        """Get raw conversation history as MemoryEntry objects"""
        if conversation_id is None:
            all_messages = []
            for conv_messages in self.conversations.values():
                all_messages.extend(conv_messages)
            return all_messages
        
        return self.conversations.get(conversation_id, [])
    
    def clear(self, conversation_id: str = None) -> None:
        """Clear conversation history"""
        if conversation_id is None:
            self.conversations.clear()
        elif conversation_id in self.conversations:
            del self.conversations[conversation_id]
        
        # Auto-save if enabled
        if self.auto_save and self.save_path:
            try:
                self.save(self.save_path)
            except Exception as e:
                self.logger.error(f"Failed to auto-save after clear: {e}")
    
    def get_conversation_ids(self) -> List[str]:
        """Get all conversation IDs"""
        return list(self.conversations.keys())
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary information about a conversation"""
        if conversation_id not in self.conversations:
            return {}
        
        messages = self.conversations[conversation_id]
        if not messages:
            return {}
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "first_message": messages[0].timestamp,
            "last_message": messages[-1].timestamp,
            "roles": list(set(msg.role for msg in messages))
        }
    
    def search_messages(self, query: str, conversation_id: str = None) -> List[MemoryEntry]:
        """Search for messages containing the query string"""
        results = []
        
        conversations_to_search = [conversation_id] if conversation_id else self.conversations.keys()
        
        for conv_id in conversations_to_search:
            if conv_id in self.conversations:
                for message in self.conversations[conv_id]:
                    if query.lower() in message.content.lower():
                        results.append(message)
        
        return results
    
    def save(self, file_path: str) -> None:
        """Save memory to JSON file"""
        try:
            # Convert MemoryEntry objects to dictionaries
            data = {}
            for conv_id, messages in self.conversations.items():
                data[conv_id] = [asdict(msg) for msg in messages]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "conversations": data,
                    "metadata": {
                        "max_history": self.max_history,
                        "saved_at": datetime.now().isoformat()
                    }
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Memory saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save memory to {file_path}: {e}")
            raise
    
    def load(self, file_path: str) -> None:
        """Load memory from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dictionaries back to MemoryEntry objects
            self.conversations = {}
            for conv_id, messages in data.get("conversations", {}).items():
                self.conversations[conv_id] = [
                    MemoryEntry(**msg) for msg in messages
                ]
            
            # Load metadata if available
            metadata = data.get("metadata", {})
            if "max_history" in metadata:
                self.max_history = metadata["max_history"]
            
            self.logger.info(f"Memory loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load memory from {file_path}: {e}")
            raise
    
    def export_conversation(self, conversation_id: str, format: str = "json") -> str:
        """Export a conversation in various formats"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = self.conversations[conversation_id]
        
        if format.lower() == "json":
            return json.dumps([asdict(msg) for msg in messages], indent=2, ensure_ascii=False)
        
        elif format.lower() == "txt":
            lines = []
            for msg in messages:
                timestamp = datetime.fromisoformat(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"[{timestamp}] {msg.role.upper()}: {msg.content}")
                if msg.tool_calls:
                    lines.append(f"  Tool calls: {json.dumps(msg.tool_calls)}")
            return "\n".join(lines)
        
        elif format.lower() == "markdown":
            lines = [f"# Conversation: {conversation_id}\n"]
            for msg in messages:
                timestamp = datetime.fromisoformat(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"## {msg.role.title()} ({timestamp})\n")
                lines.append(f"{msg.content}\n")
                if msg.tool_calls:
                    lines.append(f"**Tool calls:** ```json\n{json.dumps(msg.tool_calls, indent=2)}\n```\n")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        total_messages = sum(len(conv) for conv in self.conversations.values())
        
        role_counts = {}
        for conv in self.conversations.values():
            for msg in conv:
                role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
        
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "max_history": self.max_history,
            "role_distribution": role_counts,
            "average_messages_per_conversation": total_messages / len(self.conversations) if self.conversations else 0
        }


class PersistentMemory(ConversationMemory):
    """Memory that automatically persists to disk"""
    
    def __init__(self, save_path: str, max_history: int = 50):
        super().__init__(max_history=max_history, auto_save=True, save_path=save_path)


class MemoryManager:
    """Manager for multiple memory instances"""
    
    def __init__(self):
        self.memories: Dict[str, Memory] = {}
    
    def create_memory(self, name: str, memory_type: str = "conversation", **kwargs) -> Memory:
        """Create a new memory instance"""
        if memory_type == "conversation":
            memory = ConversationMemory(**kwargs)
        elif memory_type == "persistent":
            memory = PersistentMemory(**kwargs)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        self.memories[name] = memory
        return memory
    
    def get_memory(self, name: str) -> Optional[Memory]:
        """Get a memory instance by name"""
        return self.memories.get(name)
    
    def list_memories(self) -> List[str]:
        """List all memory names"""
        return list(self.memories.keys())
    
    def remove_memory(self, name: str) -> None:
        """Remove a memory instance"""
        if name in self.memories:
            del self.memories[name]