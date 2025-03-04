import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

# Define the path for storing chat histories
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_histories')

# Ensure the chat history directory exists
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

class ChatDatabase:
    """A simple database for managing chat histories"""
    
    @staticmethod
    def save_chat(chat_id: Optional[str] = None, title: str = None, messages: List[Dict] = None) -> str:
        """Save a chat history to disk
        
        Args:
            chat_id: Optional ID for existing chat, if None a new ID will be generated
            title: Title of the chat
            messages: List of message objects with 'role' and 'content' keys
            
        Returns:
            The chat ID
        """
        if not chat_id:
            # Generate a new chat ID based on timestamp
            chat_id = f"chat_{int(time.time())}"
            
        # If no title provided, use the first few words of the first message or a default
        if not title and messages and len(messages) > 0:
            first_msg = next((m for m in messages if m.get('role') == 'user'), None)
            if first_msg and 'content' in first_msg:
                # Use first few words of first message as title
                title = first_msg['content'][:30] + ('...' if len(first_msg['content']) > 30 else '')
            else:
                title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        elif not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
        # Create chat data structure
        chat_data = {
            'id': chat_id,
            'title': title,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'messages': messages or []
        }
        
        # Save to disk
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        with open(file_path, 'w') as f:
            json.dump(chat_data, f, indent=2)
            
        return chat_id
    
    @staticmethod
    def get_chat(chat_id: str) -> Optional[Dict]:
        """Retrieve a specific chat by ID
        
        Args:
            chat_id: The ID of the chat to retrieve
            
        Returns:
            The chat data or None if not found
        """
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def update_chat(chat_id: str, messages: List[Dict], title: str = None) -> bool:
        """Update an existing chat with new messages
        
        Args:
            chat_id: The ID of the chat to update
            messages: The new list of messages
            title: Optional new title for the chat
            
        Returns:
            True if successful, False otherwise
        """
        chat_data = ChatDatabase.get_chat(chat_id)
        if not chat_data:
            return False
            
        chat_data['messages'] = messages
        chat_data['updated_at'] = datetime.now().isoformat()
        
        if title:
            chat_data['title'] = title
            
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        with open(file_path, 'w') as f:
            json.dump(chat_data, f, indent=2)
            
        return True
    
    @staticmethod
    def add_message_to_chat(chat_id: str, role: str, content: str) -> bool:
        """Add a single message to an existing chat
        
        Args:
            chat_id: The ID of the chat to update
            role: The role of the message sender (user/assistant)
            content: The message content
            
        Returns:
            True if successful, False otherwise
        """
        chat_data = ChatDatabase.get_chat(chat_id)
        if not chat_data:
            return False
            
        chat_data['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        chat_data['updated_at'] = datetime.now().isoformat()
        
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        with open(file_path, 'w') as f:
            json.dump(chat_data, f, indent=2)
            
        return True
    
    @staticmethod
    def list_chats() -> List[Dict]:
        """List all available chats
        
        Returns:
            List of chat metadata (id, title, created_at, updated_at)
        """
        chats = []
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(CHAT_HISTORY_DIR, filename)
                try:
                    with open(file_path, 'r') as f:
                        chat_data = json.load(f)
                        chats.append({
                            'id': chat_data.get('id'),
                            'title': chat_data.get('title'),
                            'created_at': chat_data.get('created_at'),
                            'updated_at': chat_data.get('updated_at'),
                            'message_count': len(chat_data.get('messages', []))
                        })
                except Exception as e:
                    print(f"Error reading chat file {filename}: {e}")
                    
        # Sort by updated_at (most recent first)
        chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return chats
    
    @staticmethod
    def delete_chat(chat_id: str) -> bool:
        """Delete a chat by ID
        
        Args:
            chat_id: The ID of the chat to delete
            
        Returns:
            True if successful, False otherwise
        """
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        if not os.path.exists(file_path):
            return False
            
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting chat {chat_id}: {e}")
            return False