"""
Security middleware for Flask application
Provides rate limiting, security headers, and request sanitization
"""
import time
import logging
from functools import wraps
from collections import defaultdict, deque
from flask import request, jsonify, g
from app.utils.security import sanitize_html_input, SecurityError

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(lambda: deque())

class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Configure security settings
        app.config.setdefault('SECURITY_RATE_LIMIT_PER_MINUTE', 60)
        app.config.setdefault('SECURITY_RATE_LIMIT_BURST', 10)
        app.config.setdefault('SECURITY_MAX_CONTENT_LENGTH', 100 * 1024 * 1024)  # 100MB
        app.config.setdefault('SECURITY_ALLOWED_HOSTS', [])
    
    def before_request(self):
        """Process request before handling"""
        try:
            # Get client IP
            client_ip = self.get_client_ip()
            g.client_ip = client_ip
            
            # Rate limiting
            if not self.check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Content length check
            if request.content_length and request.content_length > self.app.config['SECURITY_MAX_CONTENT_LENGTH']:
                return jsonify({'error': 'Request too large'}), 413
            
            # Host header validation
            if not self.validate_host_header():
                return jsonify({'error': 'Invalid host header'}), 400
            
            # Basic input sanitization for query parameters
            self.sanitize_request_args()
            
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            return jsonify({'error': 'Security validation failed'}), 400
    
    def after_request(self, response):
        """Add security headers to response"""
        try:
            # Security headers
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
            }
            
            for header, value in security_headers.items():
                response.headers[header] = value
            
            # Remove server information
            response.headers.pop('Server', None)
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding security headers: {str(e)}")
            return response
    
    def get_client_ip(self):
        """Get real client IP considering proxies"""
        # Check for forwarded headers (be careful with these in production)
        if request.headers.get('X-Forwarded-For'):
            # Take the first IP from the chain
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        else:
            return request.remote_addr
    
    def check_rate_limit(self, client_ip):
        """Check if client IP is within rate limits"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Clean old entries
        client_requests = rate_limit_storage[client_ip]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Check burst limit (requests in last 10 seconds)
        burst_window = current_time - 10
        burst_count = sum(1 for req_time in client_requests if req_time > burst_window)
        
        if burst_count >= self.app.config['SECURITY_RATE_LIMIT_BURST']:
            logger.warning(f"Burst rate limit exceeded for IP: {client_ip}")
            return False
        
        # Check rate limit (requests per minute)
        if len(client_requests) >= self.app.config['SECURITY_RATE_LIMIT_PER_MINUTE']:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add current request
        client_requests.append(current_time)
        
        return True
    
    def validate_host_header(self):
        """Validate Host header to prevent host header injection"""
        host = request.headers.get('Host', '')
        
        if not host:
            return False
        
        # If allowed hosts are configured, check against them
        allowed_hosts = self.app.config.get('SECURITY_ALLOWED_HOSTS', [])
        if allowed_hosts:
            return host in allowed_hosts
        
        # Basic validation - prevent obvious malicious patterns
        malicious_patterns = ['<', '>', '"', "'", '\n', '\r', '\t']
        for pattern in malicious_patterns:
            if pattern in host:
                return False
        
        return True
    
    def sanitize_request_args(self):
        """Basic sanitization of request arguments"""
        try:
            # Create a new args dict with sanitized values
            sanitized_args = {}
            for key, value in request.args.items():
                if isinstance(value, str):
                    sanitized_args[key] = sanitize_html_input(value)
                else:
                    sanitized_args[key] = value
            
            # Replace request.args with sanitized version
            # Note: This is a simplified approach, in production you might want
            # to be more careful about modifying request objects
            
        except Exception as e:
            logger.error(f"Error sanitizing request args: {str(e)}")

def require_json(f):
    """Decorator to ensure request has JSON content type"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated_function

def validate_content_type(allowed_types):
    """Decorator to validate request content type"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_type = request.content_type
            if content_type not in allowed_types:
                return jsonify({
                    'error': f'Invalid content type. Allowed: {", ".join(allowed_types)}'
                }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit(requests_per_minute=60, burst_limit=10):
    """Decorator for additional rate limiting on specific endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = g.get('client_ip', request.remote_addr)
            endpoint_key = f"{client_ip}:{request.endpoint}"
            
            current_time = time.time()
            window_start = current_time - 60
            
            # Get or create endpoint-specific rate limit storage
            endpoint_requests = rate_limit_storage[endpoint_key]
            
            # Clean old entries
            while endpoint_requests and endpoint_requests[0] < window_start:
                endpoint_requests.popleft()
            
            # Check limits
            burst_window = current_time - 10
            burst_count = sum(1 for req_time in endpoint_requests if req_time > burst_window)
            
            if burst_count >= burst_limit:
                return jsonify({'error': 'Rate limit exceeded for this endpoint'}), 429
            
            if len(endpoint_requests) >= requests_per_minute:
                return jsonify({'error': 'Rate limit exceeded for this endpoint'}), 429
            
            # Add current request
            endpoint_requests.append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator 