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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
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
        
        # Get current system information for context
        system_context = get_system_context()
        
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
