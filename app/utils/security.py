"""
Security utilities for input validation, path sanitization, and file security
"""
import os
import re
import mimetypes
import hashlib
import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import magic

logger = logging.getLogger(__name__)

# Security constants
MAX_FILENAME_LENGTH = 255
MAX_PATH_DEPTH = 10
ALLOWED_PATH_CHARS = re.compile(r'^[a-zA-Z0-9_\-./\\]+$')
DANGEROUS_EXTENSIONS = {
    'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar', 'php', 'py',
    'sh', 'pl', 'rb', 'asp', 'aspx', 'jsp', 'cgi', 'dll', 'so', 'bin'
}

# Allowed MIME types for uploads
ALLOWED_IMAGE_MIMES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'
}
ALLOWED_VIDEO_MIMES = {
    'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/mkv', 'video/webm'
}
ALLOWED_MIMES = ALLOWED_IMAGE_MIMES | ALLOWED_VIDEO_MIMES

class SecurityError(Exception):
    """Security validation error"""
    pass

class PathSecurityError(SecurityError):
    """Path traversal/injection security error"""
    pass

class FileSecurityError(SecurityError):
    """File upload/validation security error"""
    pass

def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """
    Safely sanitize filename to prevent path injection and other attacks
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        str: Sanitized filename
        
    Raises:
        FileSecurityError: If filename is invalid
    """
    if not filename or not isinstance(filename, str):
        raise FileSecurityError("Invalid filename provided")
    
    # Remove null bytes and control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Use werkzeug's secure_filename as base
    safe_filename = secure_filename(filename)
    
    if not safe_filename:
        raise FileSecurityError("Filename becomes empty after sanitization")
    
    # Additional checks
    if len(safe_filename) > max_length:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:max_length-len(ext)] + ext
    
    # Check for dangerous extensions
    ext = safe_filename.rsplit('.', 1)[-1].lower() if '.' in safe_filename else ''
    if ext in DANGEROUS_EXTENSIONS:
        raise FileSecurityError(f"Dangerous file extension: .{ext}")
    
    # Prevent hidden files and special names
    if safe_filename.startswith('.') or safe_filename.lower() in ['con', 'prn', 'aux', 'nul']:
        raise FileSecurityError("Invalid filename pattern")
    
    return safe_filename

def validate_path(path: str, base_path: str) -> str:
    """
    Validate and sanitize file path to prevent directory traversal
    
    Args:
        path: File path to validate
        base_path: Base directory that path should be within
        
    Returns:
        str: Validated absolute path
        
    Raises:
        PathSecurityError: If path is invalid or outside base directory
    """
    if not path or not isinstance(path, str):
        raise PathSecurityError("Invalid path provided")
    
    # Remove null bytes and normalize
    path = path.replace('\0', '')
    
    # Convert to Path objects for safer handling
    try:
        base_path_obj = Path(base_path).resolve()
        file_path_obj = Path(path)
        
        # If path is relative, make it relative to base_path
        if not file_path_obj.is_absolute():
            file_path_obj = base_path_obj / file_path_obj
        
        # Resolve to absolute path
        resolved_path = file_path_obj.resolve()
        
    except (OSError, ValueError) as e:
        raise PathSecurityError(f"Invalid path format: {e}")
    
    # Check if resolved path is within base directory
    try:
        resolved_path.relative_to(base_path_obj)
    except ValueError:
        raise PathSecurityError("Path traversal detected - path outside allowed directory")
    
    # Check path depth
    relative_parts = resolved_path.relative_to(base_path_obj).parts
    if len(relative_parts) > MAX_PATH_DEPTH:
        raise PathSecurityError(f"Path too deep (max {MAX_PATH_DEPTH} levels)")
    
    # Check for dangerous path components
    for part in relative_parts:
        if not ALLOWED_PATH_CHARS.match(part):
            raise PathSecurityError(f"Invalid path component: {part}")
    
    return str(resolved_path)

def validate_file_upload(file: FileStorage, allowed_extensions: set = None) -> Dict[str, Any]:
    """
    Comprehensive file upload validation
    
    Args:
        file: Uploaded file object
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        dict: Validation result with file info
        
    Raises:
        FileSecurityError: If file validation fails
    """
    if not file or not file.filename:
        raise FileSecurityError("No file provided")
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    
    # Check file extension
    ext = safe_filename.rsplit('.', 1)[-1].lower() if '.' in safe_filename else ''
    if allowed_extensions and ext not in allowed_extensions:
        raise FileSecurityError(f"File extension '.{ext}' not allowed")
    
    # Read first chunk for MIME detection
    file.seek(0)
    file_header = file.read(8192)  # Read first 8KB
    file.seek(0)  # Reset file pointer
    
    if not file_header:
        raise FileSecurityError("Empty file not allowed")
    
    # Detect MIME type using file content (not just extension)
    try:
        detected_mime = magic.from_buffer(file_header, mime=True)
    except Exception as e:
        logger.warning(f"Failed to detect MIME type using magic: {e}")
        # Fallback to mimetypes
        detected_mime, _ = mimetypes.guess_type(safe_filename)
    
    if not detected_mime:
        raise FileSecurityError("Unable to determine file type")
    
    # Validate MIME type
    if detected_mime not in ALLOWED_MIMES:
        raise FileSecurityError(f"File type '{detected_mime}' not allowed")
    
    # Check for polyglot files (files that are valid in multiple formats)
    if _is_polyglot_file(file_header):
        raise FileSecurityError("Polyglot file detected - potential security risk")
    
    # Determine file category
    file_type = 'image' if detected_mime in ALLOWED_IMAGE_MIMES else 'video'
    
    return {
        'safe_filename': safe_filename,
        'detected_mime': detected_mime,
        'file_type': file_type,
        'extension': ext,
        'file_size': len(file_header)  # This is just header size, actual size calculated later
    }

def _is_polyglot_file(file_header: bytes) -> bool:
    """
    Check if file might be a polyglot (multi-format) file
    These can be security risks as they can be interpreted as different file types
    """
    # Check for common polyglot signatures
    polyglot_signatures = [
        b'\x89PNG',  # PNG
        b'\xFF\xD8\xFF',  # JPEG
        b'GIF8',  # GIF
        b'RIFF',  # Various formats including AVI
        b'\x00\x00\x00\x20ftypM4V',  # MP4
    ]
    
    # Count how many different format signatures are found
    matches = 0
    for sig in polyglot_signatures:
        if sig in file_header[:512]:  # Check first 512 bytes
            matches += 1
    
    return matches > 1

def validate_request_params(params: Dict[str, Any], validation_rules: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate request parameters against rules
    
    Args:
        params: Dictionary of parameters to validate
        validation_rules: Dictionary of validation rules per parameter
        
    Returns:
        dict: Validated and sanitized parameters
        
    Raises:
        SecurityError: If validation fails
    """
    validated = {}
    
    for param_name, rules in validation_rules.items():
        value = params.get(param_name)
        
        # Check required parameters
        if rules.get('required', False) and value is None:
            raise SecurityError(f"Required parameter '{param_name}' missing")
        
        if value is None:
            validated[param_name] = rules.get('default')
            continue
        
        # Type validation
        expected_type = rules.get('type')
        if expected_type:
            if expected_type == 'int':
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise SecurityError(f"Parameter '{param_name}' must be an integer")
            elif expected_type == 'float':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise SecurityError(f"Parameter '{param_name}' must be a number")
            elif expected_type == 'bool':
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = bool(value)
            elif expected_type == 'str':
                value = str(value)
        
        # Range validation
        if 'min' in rules and value < rules['min']:
            raise SecurityError(f"Parameter '{param_name}' must be >= {rules['min']}")
        if 'max' in rules and value > rules['max']:
            raise SecurityError(f"Parameter '{param_name}' must be <= {rules['max']}")
        
        # String length validation
        if isinstance(value, str):
            if 'min_length' in rules and len(value) < rules['min_length']:
                raise SecurityError(f"Parameter '{param_name}' too short (min {rules['min_length']})")
            if 'max_length' in rules and len(value) > rules['max_length']:
                raise SecurityError(f"Parameter '{param_name}' too long (max {rules['max_length']})")
        
        # Pattern validation
        if 'pattern' in rules and isinstance(value, str):
            if not re.match(rules['pattern'], value):
                raise SecurityError(f"Parameter '{param_name}' has invalid format")
        
        # Allowed values validation
        if 'allowed' in rules and value not in rules['allowed']:
            raise SecurityError(f"Parameter '{param_name}' must be one of: {rules['allowed']}")
        
        validated[param_name] = value
    
    return validated

def sanitize_html_input(text: str) -> str:
    """
    Basic HTML sanitization to prevent XSS
    For production, consider using a proper library like bleach
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    return text.strip()

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token
    """
    import secrets
    return secrets.token_urlsafe(length)

def hash_file_content(file_path: str) -> str:
    """
    Generate SHA-256 hash of file content for integrity checking
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        return ""

def validate_json_input(data: Any, max_depth: int = 10, max_size: int = 1024*1024) -> Any:
    """
    Validate JSON input to prevent JSON bombs and deep nesting attacks
    """
    import json
    
    # Convert to JSON string to check size
    try:
        json_str = json.dumps(data)
        if len(json_str) > max_size:
            raise SecurityError(f"JSON payload too large (max {max_size} bytes)")
    except TypeError:
        raise SecurityError("Invalid JSON data")
    
    # Check nesting depth
    def check_depth(obj, current_depth=0):
        if current_depth > max_depth:
            raise SecurityError(f"JSON nesting too deep (max {max_depth})")
        
        if isinstance(obj, dict):
            for value in obj.values():
                check_depth(value, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                check_depth(item, current_depth + 1)
    
    check_depth(data)
    return data 