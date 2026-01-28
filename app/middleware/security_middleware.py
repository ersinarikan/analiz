"""
Security middleware for Flask application
Provides rate limiting, security headers, and request sanitization
"""
import time 
import logging 
from functools import wraps 
from collections import defaultdict ,deque 
from flask import request ,jsonify ,g 
from app .utils .security import sanitize_html_input 

logger =logging .getLogger (__name__ )

# ERSIN Rate limiting storage (in production, use Redis)
rate_limit_storage =defaultdict (lambda :deque ())

class SecurityMiddleware :
    """Security middleware for request processing"""

    def __init__ (self ,app =None ):
        self .app =app 
        if app is not None :
            self .init_app (app )

    def init_app (self ,app ):
        """Initialize security middleware with Flask app"""
        app .before_request (self .before_request )
        app .after_request (self .after_request )

        # ERSIN Configure security settings - More permissive için development
        app .config .setdefault ('SECURITY_RATE_LIMIT_PER_MINUTE',500 )# ERSIN Increased from 200 için analysis operations
        app .config .setdefault ('SECURITY_RATE_LIMIT_BURST',100 )# ERSIN Increased from 50 için analysis operations
        app .config .setdefault ('SECURITY_MAX_CONTENT_LENGTH',100 *1024 *1024 )# ERSIN 100MB
        app .config .setdefault ('SECURITY_ALLOWED_HOSTS',[])

    def before_request (self ):
        """Process request before handling"""
        try :
        # ERSIN Get client IP
            client_ip =self .get_client_ip ()
            g .client_ip =client_ip 

            # ERSIN Skip rate limiting for localhost in development and for file uploads
            if self .should_skip_rate_limiting ():
                return None 

                # ERSIN Rate limiting
            if not self .check_rate_limit (client_ip ):
                return jsonify ({'error':'Rate limit exceeded'}),429 

                # ERSIN Content length check
            if self .app is not None and request .content_length and request .content_length >self .app .config ['SECURITY_MAX_CONTENT_LENGTH']:
                return jsonify ({'error':'Request too large'}),413 

                # ERSIN Host header validation
            if not self .validate_host_header ():
                return jsonify ({'error':'Invalid host header'}),400 

                # ERSIN Basic input sanitization için query parameters
            self .sanitize_request_args ()

        except Exception as e :
            logger .error (f"Security middleware error: {str (e )}")
            return jsonify ({'error':'Security validation failed'}),400 

    def after_request (self ,response ):
        """Add security headers to response"""
        try :
        # ERSIN Security headers
            security_headers ={
            'X-Content-Type-Options':'nosniff',
            'X-Frame-Options':'DENY',
            'X-XSS-Protection':'1; mode=block',
            'Strict-Transport-Security':'max-age=31536000; includeSubDomains',
            'Content-Security-Policy':"default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.socket.io https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' https://analiz.aile.gov.tr ws: wss: https:;",
            'Referrer-Policy':'strict-origin-when-cross-origin',
            'Permissions-Policy':'geolocation=(), microphone=(), camera=()'
            }

            for header ,value in security_headers .items ():
                response .headers [header ]=value 

                # ERSIN Remove server information
            response .headers .pop ('Server',None )

            return response 

        except Exception as e :
            logger .error (f"Error adding security headers: {str (e )}")
            return response 

    def get_client_ip (self ):
        """Get real client IP considering proxies"""
        # ERSIN Check için forwarded headers (be careful ile these in production)
        x_forwarded_for =request .headers .get ('X-Forwarded-For')
        if x_forwarded_for :
        # ERSIN Take the first IP from the chain
            return x_forwarded_for .split (',')[0 ].strip ()
        elif request .headers .get ('X-Real-IP'):
            return request .headers .get ('X-Real-IP')
        else :
            return request .remote_addr 

    def check_rate_limit (self ,client_ip ):
        """Check if client IP is within rate limits"""
        current_time =time .time ()
        window_start =current_time -60 # ERSIN 1 minute window

        # ERSIN Clean old entries
        client_requests =rate_limit_storage [client_ip ]
        while client_requests and client_requests [0 ]<window_start :
            client_requests .popleft ()

            # ERSIN Check burst limit (requests in last 10 seconds)
        burst_window =current_time -10 
        burst_count =sum (1 for req_time in client_requests if req_time >burst_window )

        if self .app is not None and burst_count >=self .app .config ['SECURITY_RATE_LIMIT_BURST']:
            logger .warning (f"Burst rate limit exceeded for IP: {client_ip }")
            return False 

            # ERSIN Check rate limit (requests per minute)
        if self .app is not None and len (client_requests )>=self .app .config ['SECURITY_RATE_LIMIT_PER_MINUTE']:
            logger .warning (f"Rate limit exceeded for IP: {client_ip }")
            return False 

            # ERSIN Add current request
        client_requests .append (current_time )

        return True 

    def validate_host_header (self ):
        """Validate Host header to prevent host header injection"""
        host =request .headers .get ('Host','')

        if not host :
            return False 

            # ERSIN If allowed hosts yapılandırılmıştır, check against them
        if self .app is None :
            return True 
        allowed_hosts =self .app .config .get ('SECURITY_ALLOWED_HOSTS',[])
        if allowed_hosts :
            return host in allowed_hosts 

            # ERSIN Basic validation - önlemek obvious malicious patterns
        malicious_patterns =['<','>','"',"'",'\n','\r','\t']
        for pattern in malicious_patterns :
            if pattern in host :
                return False 

        return True 

    def sanitize_request_args (self ):
        """Basic sanitization of request arguments"""
        try :
        # ERSIN Create  new args dict ile sanitized values
            sanitized_args ={}
            for key ,value in request .args .items ():
                if isinstance (value ,str ):
                    sanitized_args [key ]=sanitize_html_input (value )
                else :
                    sanitized_args [key ]=value 

                    # ERSIN Replace request.args ile sanitized version
                    # ERSIN Not: Bu basitleştirilmiş yaklaşım, production'da request object'lerini değiştirirken daha dikkatli olmak gerekebilir

        except Exception as e :
            logger .error (f"Error sanitizing request args: {str (e )}")

    def should_skip_rate_limiting (self ):
        """Determine if rate limiting should be skipped for this request"""
        # ERSIN Skip için localhost in development
        if request .remote_addr in ['127.0.0.1','::1','localhost']:
        # ERSIN Skip rate limiting for file upload endpoints
            if '/api/files/'in request .path :
                return True 

                # ERSIN Skip için Socket.IO connections
            if request .path .startswith ('/socket.io/'):
                return True 

                # ERSIN Skip for static files
            if request .path .startswith ('/static/'):
                return True 

                # ERSIN Skip için analysis status checks to prevent polling issues
            if '/api/analysis/'in request .path and '/status'in request .path :
                return True 

                # ERSIN Skip için debug queue status checks
            if '/api/debug/queue-status'in request .path :
                return True 

        return False 

def require_json (f ):
    """Decorator to ensure request has JSON content type"""
    @wraps (f )
    def decorated_function (*args ,**kwargs ):
        if not request .is_json :
            return jsonify ({'error':'Content-Type must be application/json'}),400 
        return f (*args ,**kwargs )
    return decorated_function 

def validate_content_type (allowed_types ):
    """Decorator to validate request content type"""
    def decorator (f ):
        @wraps (f )
        def decorated_function (*args ,**kwargs ):
            content_type =request .content_type 
            if content_type not in allowed_types :
                return jsonify ({
                'error':f'Invalid content type. Allowed: {", ".join (allowed_types )}'
                }),400 
            return f (*args ,**kwargs )
        return decorated_function 
    return decorator 

def rate_limit (requests_per_minute =60 ,burst_limit =10 ):
    """Decorator for additional rate limiting on specific endpoints"""
    def decorator (f ):
        @wraps (f )
        def decorated_function (*args ,**kwargs ):
            client_ip =g .get ('client_ip',request .remote_addr )
            endpoint_key =f"{client_ip }:{request .endpoint }"

            # ERSIN Skip rate limiting for localhost file uploads
            if client_ip in ['127.0.0.1','::1']and '/api/files/'in request .path :
                return f (*args ,**kwargs )

            current_time =time .time ()
            window_start =current_time -60 

            # ERSIN Get veya create endpoint-specific rate limit storage
            endpoint_requests =rate_limit_storage [endpoint_key ]

            # ERSIN Clean old entries
            while endpoint_requests and endpoint_requests [0 ]<window_start :
                endpoint_requests .popleft ()

                # ERSIN Check limits
            burst_window =current_time -10 
            burst_count =sum (1 for req_time in endpoint_requests if req_time >burst_window )

            if burst_count >=burst_limit :
                return jsonify ({'error':'Rate limit exceeded for this endpoint'}),429 

            if len (endpoint_requests )>=requests_per_minute :
                return jsonify ({'error':'Rate limit exceeded for this endpoint'}),429 

                # ERSIN Add current request
            endpoint_requests .append (current_time )

            return f (*args ,**kwargs )
        return decorated_function 
    return decorator 