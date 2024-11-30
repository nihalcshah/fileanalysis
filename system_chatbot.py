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

app = Flask(__name__)

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

def get_file_info(path: str = "/Users/nihalshah/Work/Personal/Private Analysis") -> Dict:
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

def search_files(query: str, path: str = "/Users/nihalshah/Work/Personal/Private Analysis") -> List[Dict]:
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
    except Exception as e:
        print(f"Error searching files: {e}")
    
    return results

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
    
    if mime_type is None:
        # Try to read as text by default
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            # If can't read as text, treat as binary/image
            mime_type = 'application/octet-stream'
    
    if mime_type and mime_type.startswith('text/'):
        with open(file_path, 'r') as f:
            return f.read()
    else:
        # Handle as image using Ollama
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = ollama.chat(
            model='llama3.2',
            messages=[{
                'role': 'user',
                'content': 'Describe this image in detail',
                'images': [image_data]
            }]
        )
        return response['message']['content']

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files')
def list_files():
    path = request.args.get('path', '/Users/nihalshah/Work/Personal/Private Analysis')
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
    if not query:
        return jsonify([])
    return jsonify(search_files(query))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        selected_file = data.get('selected_file')
        stream = data.get('stream', False)
        
        context = get_system_context()
        if selected_file:
            try:
                file_content = process_file(selected_file)
                context += f"\n\nSelected file content:\n{file_content}"
            except Exception as e:
                print(f"Error processing file: {e}")
        
        if stream:
            def generate():
                for chunk in ollama.chat(
                    model='llama3.2',
                    messages=[
                        {'role': 'system', 'content': context},
                        {'role': 'user', 'content': message}
                    ],
                    stream=True
                ):
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield f"data: {json.dumps({'response': chunk['message']['content']})}\n\n"
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': message}
            ]
        )
        
        return jsonify({"response": response['message']['content']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
