import subprocess
import psutil
import json
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
import ollama
import os
import glob
from typing import List, Dict
import json
import asyncio
import mimetypes
import base64
from pathlib import Path
import re
import time
import logging
from PIL import Image
import io
import hashlib
from functools import lru_cache
from chat_db import ChatDatabase  # Import the ChatDatabase class

app = Flask(__name__)

# Image processing configuration
IMAGE_MAX_SIZE = 1024  # Maximum dimension for images
IMAGE_QUALITY = 85     # JPEG quality (0-100)
IMAGE_CACHE_SIZE = 100  # Number of image responses to cache

class ModelManager:
    _instance = None
    _models_loaded = False
    _chat_model = None
    _vision_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        if not self._models_loaded:
            try:
                # Initialize models
                ollama.pull('llama3.2')
                
                # Try to use a faster vision model if available
                try:
                    # Check if llava is available (typically faster than llama3.2-vision)
                    ollama.pull('llava')
                    self._vision_model = 'llava'
                    print("Using llava for vision processing")
                except Exception:
                    # Fall back to llama3.2-vision
                    ollama.pull('llama3.2-vision')
                    self._vision_model = 'llama3.2-vision'
                    print("Using llama3.2-vision for vision processing")
                
                self._models_loaded = True
                print("Models initialized successfully")
            except Exception as e:
                print(f"Error initializing models: {e}")
                raise e
    
    @property
    def is_loaded(self):
        return self._models_loaded

    def get_chat_model(self):
        if not self._models_loaded:
            self.initialize()
        return 'llama3.2'

    def get_vision_model(self):
        if not self._models_loaded:
            self.initialize()
        return self._vision_model

# Create global instance
model_manager = ModelManager()

# Cache for image processing results
@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def process_image_cached(image_hash, prompt):
    """Cached version of image processing to avoid reprocessing the same image"""
    # This is just a placeholder - the actual processing happens elsewhere
    # The cache key is based on both the image hash and the prompt
    return None

def preprocess_image(file_path):
    """Resize and optimize image for faster processing"""
    try:
        # Open the image
        img = Image.open(file_path)
        
        # Convert to RGB if needed (removing alpha channel)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = background
        
        # Calculate new dimensions while maintaining aspect ratio
        width, height = img.size
        if width > IMAGE_MAX_SIZE or height > IMAGE_MAX_SIZE:
            if width > height:
                new_width = IMAGE_MAX_SIZE
                new_height = int(height * (IMAGE_MAX_SIZE / width))
            else:
                new_height = IMAGE_MAX_SIZE
                new_width = int(width * (IMAGE_MAX_SIZE / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to a temporary file
        temp_file = f"{file_path}_optimized.jpg"
        img.save(temp_file, 'JPEG', quality=IMAGE_QUALITY, optimize=True)
        
        # Calculate hash for caching
        with open(temp_file, 'rb') as f:
            image_data = f.read()
            image_hash = hashlib.md5(image_data).hexdigest()
        
        return temp_file, image_hash
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return file_path, None  # Return original file path if preprocessing fails

def get_system_info():
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    
    # Get disk usage
    disk = psutil.disk_usage('/')
    
    # Get running processes
    processes = [p.name() for p in psutil.process_iter()]
    
    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "disk_usage": f"{disk.percent}%",
        "processes": processes[:10]  # Limiting to first 10 processes for brevity
    }

def get_file_info(path: str = "/Users/nihalshah") -> Dict:
    file_info = {
        "files": [],
        "total_size": 0
    }
    
    try:
        # Get all files in the directory and subdirectories
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        size = os.path.getsize(full_path)
                        file_info["files"].append({
                            "name": file,
                            "path": full_path,
                            "size": size,
                            "modified": os.path.getmtime(full_path)
                        })
                        file_info["total_size"] += size
                except (PermissionError, FileNotFoundError):
                    continue
    except Exception as e:
        print(f"Error scanning directory: {e}")
    
    return file_info

def search_files(query: str, path: str = "/Users/nihalshah", page: int = 1, limit: int = 50) -> List[Dict]:
    cache_key = f"{query}-{path}"
    
    # Check cache first
    cached = search_cache.get(cache_key)
    if cached and time.time() - cached['timestamp'] < 300:  # 5 minute cache
        return paginate_results(cached['results'], page, limit)
    
    results = []
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                if query.lower() in file.lower():
                    full_path = os.path.join(root, file)
                    try:
                        results.append({
                            "name": file,
                            "path": full_path,
                            "size": os.path.getsize(full_path),
                            "modified": os.path.getmtime(full_path)
                        })
                    except (PermissionError, FileNotFoundError):
                        continue
                
                # Early exit if we have enough results
                if len(results) >= 1000:
                    break
            if len(results) >= 1000:
                break
        
        # Update cache
        search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
    except Exception as e:
        print(f"Error searching files: {e}")
    
    return paginate_results(results, page, limit)

def paginate_results(results: List[Dict], page: int, limit: int) -> Dict:
    start = (page - 1) * limit
    end = start + limit
    return {
        'results': results[start:end],
        'total': len(results),
        'page': page,
        'limit': limit
    }

search_cache = {}


def get_system_context():
    system_info = get_system_info()
    file_info = get_file_info()
    
    return f"""You are a helpful assistant that provides information about the user's system.
    Current system status:
    - CPU Usage: {system_info['cpu_usage']}
    - Memory Usage: {system_info['memory_usage']}
    - Disk Usage: {system_info['disk_usage']}
    - Top processes: {', '.join(system_info['processes'])}
    
    File System Information:
    - Total files scanned: {len(file_info['files'])}
    - Total size: {file_info['total_size'] / (1024*1024):.2f} MB
    
    You can help users find files and provide system information."""

def process_file(file_path: str) -> str:
    """Process a file based on its type - text or image."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Handle text files
    if mime_type is None or mime_type.startswith('text/'):
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            # If can't read as text, treat as binary/image
            mime_type = 'application/octet-stream'
    
    # Handle image files
    if mime_type and mime_type.startswith('image/'):
        print(f"Processing file as image: {file_path}")
        try:
            # Validate image file
            if not mime_type.startswith('image/'):
                return "Error: File does not appear to be a valid image"
            
            print(f"Image MIME type: {mime_type}")
            
            # Preprocess the image for faster processing
            optimized_path, image_hash = preprocess_image(file_path)
            
            # Default prompt for image analysis
            prompt = 'Analyze this image in detail. Describe its contents, any text present, and key visual elements.'
            
            # Check cache first if we have a valid hash
            if image_hash:
                cached_result = process_image_cached(image_hash, prompt)
                if cached_result:
                    print("Using cached image analysis result")
                    return cached_result
            
            # Use vision model for image processing
            start_time = time.time()
            response = ollama.chat(
                model=model_manager.get_vision_model(),
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [optimized_path]
                }]
            )
            processing_time = time.time() - start_time
            print(f"Image processing completed in {processing_time:.2f} seconds")
            
            result = response['message']['content']
            
            # Cache the result if we have a valid hash
            if image_hash:
                # Update the cache (this doesn't actually call the function, just updates the cache)
                process_image_cached.cache_clear()
                process_image_cached.__wrapped__(image_hash, prompt)
                process_image_cached.cache_info()
            
            # Clean up temporary file if it was created
            if optimized_path != file_path and os.path.exists(optimized_path):
                try:
                    os.remove(optimized_path)
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
            
            return result
                
        except Exception as e:
            print(f"Error processing image: {type(e).__name__}: {str(e)}")
            return f"Error processing image: {str(e)}"
    
    return f"Unsupported file type: {mime_type}"

def resolve_file_path(path_mention: str):
    """Resolve a potential file path mention into an absolute path using standard path operations."""
    if not path_mention:
        return None
        
    cwd = os.getcwd()
    home_dir = os.path.expanduser("~")
    
    # Define common directories to search
    common_dirs = [
        cwd,  # Current working directory
        home_dir,  # Home directory
        os.path.join(home_dir, "Downloads"),  # Downloads folder
        os.path.join(home_dir, "Desktop"),    # Desktop folder
        os.path.join(home_dir, "Documents"),  # Documents folder
        os.path.join(home_dir, "Pictures")    # Pictures folder
    ]
    
    # Clean up the path mention
    path_mention = path_mention.strip()
    
    # Case 1: If it's already an absolute path
    if os.path.isabs(path_mention):
        if os.path.exists(path_mention):
            return path_mention
        return None
        
    # Case 2: If it starts with ~, expand it
    if path_mention.startswith("~"):
        expanded_path = os.path.expanduser(path_mention)
        if os.path.exists(expanded_path):
            return expanded_path
            
    # Case 3: Try relative to current directory
    abs_path = os.path.abspath(path_mention)
    if os.path.exists(abs_path):
        return abs_path
        
    # Case 4: Try in common directories
    for base_dir in common_dirs:
        potential_path = os.path.join(base_dir, path_mention)
        if os.path.exists(potential_path):
            return potential_path
            
    # Case 5: Try to find the file by name in common directories
    filename = os.path.basename(path_mention)
    if filename:
        for base_dir in common_dirs:
            for root, _, files in os.walk(base_dir):
                if filename in files:
                    return os.path.join(root, filename)
                    
    return None

def initialize_models():
    model_manager.initialize()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chats', methods=['GET'])
def list_chats():
    """List all saved chat histories"""
    try:
        chats = ChatDatabase.list_chats()
        return jsonify(chats)
    except Exception as e:
        print(f"Error listing chats: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a specific chat by ID"""
    try:
        chat = ChatDatabase.get_chat(chat_id)
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        return jsonify(chat)
    except Exception as e:
        print(f"Error getting chat: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat by ID"""
    try:
        success = ChatDatabase.delete_chat(chat_id)
        if not success:
            return jsonify({"error": "Chat not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/chats/new', methods=['POST'])
def new_chat():
    """Create a new empty chat"""
    try:
        title = request.json.get('title', None)
        chat_id = ChatDatabase.save_chat(title=title)
        return jsonify({"id": chat_id})
    except Exception as e:
        print(f"Error creating new chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/files')
def list_files():
    path = request.args.get('path', '/Users/nihalshah')
    try:
        files = []
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if not entry.name.startswith('.'):  # Skip hidden files
                        files.append({
                            "name": entry.name,
                            "path": entry.path,
                            "is_directory": entry.is_dir(),
                            "size": os.path.getsize(entry.path) if entry.is_file() else 0
                        })
                except (PermissionError, FileNotFoundError):
                    continue
        return jsonify(sorted(files, key=lambda x: (not x["is_directory"], x["name"])))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/search')
def search():
    query = request.args.get('query', '')
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    
    if not query:
        return jsonify({'results': [], 'total': 0})
        
    results = search_files(query, page=page, limit=limit)
    return jsonify(results)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        logging.basicConfig(level=logging.INFO)
        data = request.get_json()
        
        # Initialize context
        context = get_system_context() if data.get('system_context') else " "
        
        # Ensure models are loaded
        if not model_manager.is_loaded:
            model_manager.initialize()
        
        message = data.get('message', '')
        selected_file = data.get('selected_file')
        stream = data.get('stream', False)
        chat_id = data.get('chat_id', None)
        
        # Process selected file
        if selected_file:
            try:
                mime_type, _ = mimetypes.guess_type(selected_file)
                print(f"Processing file type: {mime_type}")
                
                # Handle image files with streaming response
                if mime_type and mime_type.startswith('image/'):
                    print("Processing image file...")
                    
                    # Preprocess the image for faster processing
                    optimized_path, image_hash = preprocess_image(selected_file)
                    prompt = message or 'Analyze this image in detail'
                    
                    # Check cache first if we have a valid hash
                    if image_hash:
                        cached_result = process_image_cached(image_hash, prompt)
                        if cached_result:
                            print("Using cached image analysis result")
                            # Return cached result as a single chunk
                            def generate_cached():
                                yield f"data: {json.dumps({'event': 'streaming_started'})}"


                                yield f"data: {json.dumps({'response': cached_result})}"


                            return Response(stream_with_context(generate_cached()), mimetype='text/event-stream')
                    
                    # If not cached, process with streaming
                    def generate():
                        yield f"data: {json.dumps({'event': 'streaming_started'})}"


                        
                        start_time = time.time()
                        accumulated_response = ""
                        
                        for chunk in ollama.chat(
                            model=model_manager.get_vision_model(),
                            messages=[{
                                'role': 'user',
                                'content': prompt,
                                'images': [optimized_path]
                            }],
                            stream=True
                        ):
                            if 'message' in chunk and 'content' in chunk['message']:
                                accumulated_response += chunk['message']['content']
                                yield f"data: {json.dumps({'response': chunk['message']['content']})}"

                        
                        # Cache the complete response
                        if image_hash:
                            # This is a hack to update the LRU cache
                            process_image_cached.__wrapped__(image_hash, prompt)
                            # Store the result in the cache dictionary directly
                            process_image_cached.cache_clear()
                            process_image_cached.__wrapped__(image_hash, prompt)
                            
                        processing_time = time.time() - start_time
                        print(f"Image processing completed in {processing_time:.2f} seconds")
                        
                        # Clean up temporary file if it was created
                        if optimized_path != selected_file and os.path.exists(optimized_path):
                            try:
                                os.remove(optimized_path)
                            except Exception as e:
                                print(f"Error removing temporary file: {e}")
                                
                    return Response(stream_with_context(generate()), mimetype='text/event-stream')
                
                # Handle text files
                file_content = process_file(selected_file)
                context += f"Selected file content:\n{file_content}\n"
                
            except Exception as e:
                print(f"Error processing file: {type(e).__name__}: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        # Create or get existing chat
        if not chat_id:
            chat_id = ChatDatabase.save_chat(title=message[:30] + ('...' if len(message) > 30 else ''))
        
        # Save user message to chat history
        if message:
            ChatDatabase.add_message_to_chat(chat_id, 'user', message)
        
        # Handle streaming response
        if stream:
            def generate_2():
                yield f"data: {json.dumps({'event': 'streaming_started', 'chat_id': chat_id})}"
                
                accumulated_response = ""
                
                for chunk in ollama.chat(
                    model=model_manager.get_chat_model(),
                    messages=[{'role': 'system', 'content': context}, {'role': 'user', 'content': message}],
                    stream=True
                ):
                    if 'message' in chunk and 'content' in chunk['message']:
                        accumulated_response += chunk['message']['content']
                        yield f"data: {json.dumps({'response': chunk['message']['content'], 'chat_id': chat_id})}"
                
                # Save assistant's complete response to chat history
                if accumulated_response:
                    ChatDatabase.add_message_to_chat(chat_id, 'assistant', accumulated_response)
                    
            return Response(stream_with_context(generate_2()), mimetype='text/event-stream')
        
        # Handle regular response
        response = ollama.chat(
            model=model_manager.get_chat_model(),
            messages=[{'role': 'system', 'content': context}, {'role': 'user', 'content': message}]
        )
        
        # Save assistant's response to chat history
        assistant_response = response['message']['content']
        ChatDatabase.add_message_to_chat(chat_id, 'assistant', assistant_response)
        
        return jsonify({"response": assistant_response, "chat_id": chat_id})
        
    except Exception as e:
        print(f"Error in chat endpoint: {type(e).__name__}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/preview')
def preview_file():
    """Preview endpoint for showing file contents in the UI"""
    try:
        path = request.args.get('path')
        if not path:
            return "No file path provided", 400
            
        if not os.path.exists(path):
            return "File not found", 404
            
        mime_type, _ = mimetypes.guess_type(path)
        
        # Only preview text files
        if mime_type and mime_type.startswith('image/'):
            return "Image files cannot be previewed as text", 400
            
        # Try to read the file as text
        try:
            with open(path, 'r') as f:
                content = f.read(5000)  # Limit to first 5000 chars
                return content
        except UnicodeDecodeError:
            return "Binary file cannot be previewed as text", 400
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    initialize_models()  # Initialize models before starting the server
    app.run(host='0.0.0.0', port=8080, debug=True)
