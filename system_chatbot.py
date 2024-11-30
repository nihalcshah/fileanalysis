import subprocess
import psutil
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import uvicorn
import os
import glob
from typing import List, Dict
from fastapi.responses import StreamingResponse
import json
import asyncio
import mimetypes
import base64
from pathlib import Path
import re

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    stream: bool = False  # Add streaming option

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

def search_files(query: str, path: str = "/Users/nihalshah") -> List[Dict]:
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
    
    You can help users find files and provide system information. For file searches, use the search_files() function."""

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
            model='llava',  # or any other Ollama model that supports image processing
            messages=[{
                'role': 'user',
                'content': 'Describe this image in detail',
                'images': [image_data]
            }]
        )
        return response['message']['content']

def resolve_file_path(path_mention: str) -> str:
    """Use LLM to resolve a potential file path mention into an absolute path."""
    # Get the current working directory for context
    cwd = os.getcwd()
    home_dir = os.path.expanduser("~")
    
    # Ask Ollama to help resolve the path
    response = ollama.chat(
        model='mistral',  # Using mistral as it's good at reasoning
        messages=[{
            'role': 'user',
            'content': f"""Given the following context:
- Current working directory: {cwd}
- Home directory: {home_dir}
- Mentioned path or file reference: "{path_mention}"

Convert this into an absolute file path. If it's already absolute, verify it. If it's relative, make it absolute.
Only respond with the absolute path, nothing else. If you can't determine a valid path, respond with 'None'."""
        }]
    )
    
    resolved_path = response['message']['content'].strip()
    
    # If LLM couldn't resolve it or says None, return None
    if resolved_path.lower() == 'none':
        return None
        
    # Clean up the path (remove quotes if present)
    resolved_path = resolved_path.strip('"\'')
    
    # Expand user directory if present
    resolved_path = os.path.expanduser(resolved_path)
    
    # Convert to absolute path if it's not already
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.abspath(resolved_path)
        
    return resolved_path

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Extract potential file path mentions from the message
        message = request.message
        words = message.split()
        file_paths = []
        
        # Look for potential file paths or references in the message
        for word in words:
            # Try to resolve the path
            resolved_path = resolve_file_path(word)
            if resolved_path and os.path.exists(resolved_path):
                file_paths.append(resolved_path)
                # Replace the original mention with a placeholder
                message = message.replace(word, f"[Content of {os.path.basename(resolved_path)}]")
        
        # Process each resolved file
        file_contents = []
        for file_path in file_paths:
            try:
                content = process_file(file_path)
                file_contents.append(f"Content of {os.path.basename(file_path)}:\n{content}")
            except Exception as e:
                file_contents.append(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        
        # Add file contents to the system context
        system_context = get_system_context()
        if file_contents:
            system_context += "\n\nProcessed Files:\n" + "\n\n".join(file_contents)

        # Extract any file search queries from the message
        message = request.message.lower()
        file_results = None
        
        if "find file" in message or "search file" in message or "look for file" in message:
            # Extract the search query - this is a simple implementation
            search_terms = message.split()
            query = search_terms[-1]  # Take the last word as the search term
            file_results = search_files(query)
            
            # Add file results to the message
            if file_results:
                request.message += f"\n\nFound these files:\n" + \
                    "\n".join([f"- {f['name']} ({f['path']})" for f in file_results[:5]])
            else:
                request.message += f"\n\nNo files found matching '{query}'"
        
        if request.stream:
            return StreamingResponse(
                stream_response(system_context, request.message),
                media_type='text/plain'  # Changed to plain text
            )
        
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': system_context
            },
            {
                'role': 'user',
                'content': request.message
            }
        ])
        
        return {"response": response['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(system_context: str, message: str):
    try:
        stream = ollama.chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': system_context
                },
                {
                    'role': 'user',
                    'content': message
                }
            ],
            stream=True
        )

        for chunk in stream:
            if 'message' in chunk:
                content = chunk['message'].get('content', '')
                if content:
                    yield content
                    await asyncio.sleep(0.01)  # Reduced delay for smoother output
        
        # Send end of stream marker
        yield "\n[DONE]"
    except Exception as e:
        yield f"\nError: {str(e)}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
