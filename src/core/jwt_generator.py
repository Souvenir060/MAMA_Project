#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - JWT Token Generator

This module provides comprehensive JWT token generation and verification
for Milestone authentication within the MAMA multi-agent system.

Academic Features:
- Cryptographic token generation with configurable algorithms
- Multi-layer security with custom claims and permissions
- Token lifecycle management with refresh capabilities
- Integration with MAMA trust-aware authentication protocols
"""

import jwt
import json
import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

# Configure comprehensive logging
logger = logging.getLogger(__name__)


class TokenType(Enum):
    """JWT token types for different MAMA system components"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    SERVICE_TOKEN = "service_token"
    AGENT_TOKEN = "agent_token"
    MILESTONE_TOKEN = "milestone_token"


class PermissionLevel(Enum):
    """Permission levels for token authorization"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SYSTEM = "system"
    AGENT_COORDINATOR = "agent_coordinator"


@dataclass
class TokenClaims:
    """Structured token claims for MAMA system authentication"""
    subject: str
    audience: str
    issuer: str = "mama_flight_assistant"
    scope: List[str] = None
    permissions: List[str] = None
    agent_capabilities: List[str] = None
    trust_level: float = 0.8
    security_context: Dict[str, Any] = None


class MilestoneTokenGenerator:
    """
    Advanced JWT Token Generator for Milestone API Authentication
    
    Provides cryptographically secure token generation with academic-level
    security features for multi-agent system authentication and authorization.
    
    Features:
    - Multiple token types with different security levels
    - Configurable cryptographic algorithms
    - Trust-aware permission management
    - Token lifecycle management with refresh capabilities
    - Integration with MAMA agent trust evaluation
    """
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 algorithm: str = "HS256",
                 default_expiry_hours: int = 24):
        """
        Initialize the JWT token generator with security configuration
        
        Args:
            secret_key: Cryptographic key for token signing (auto-generated if None)
            algorithm: JWT signing algorithm (HS256, HS512, RS256, etc.)
            default_expiry_hours: Default token expiration time in hours
        """
        self.secret_key = secret_key or self._generate_secure_key()
        self.algorithm = algorithm
        self.default_expiry_hours = default_expiry_hours
        
        # Token validation parameters
        self.clock_skew_tolerance = timedelta(seconds=30)
        self.max_token_lifetime = timedelta(days=30)
        
        # Security audit tracking
        self.token_generation_count = 0
        self.token_verification_count = 0
        self.failed_verification_count = 0
        
        logger.info(f"Milestone token generator initialized with algorithm: {algorithm}")
    
    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure signing key"""
        secure_bytes = secrets.token_bytes(64)
        return hashlib.sha256(secure_bytes).hexdigest()
    
    def generate_token(self, 
                      token_claims: TokenClaims,
                      token_type: TokenType = TokenType.ACCESS_TOKEN,
                      expires_in_hours: Optional[int] = None,
                      custom_claims: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive JWT token with academic-level security
        
        Args:
            token_claims: Structured claims for token authentication
            token_type: Type of token being generated
            expires_in_hours: Custom expiration time (uses default if None)
            custom_claims: Additional custom claims for specialized use cases
            
        Returns:
            Signed JWT token string
            
        Raises:
            ValueError: If token claims are invalid
            RuntimeError: If token generation fails
        """
        try:
            # Validate input parameters
            self._validate_token_claims(token_claims)
            
            current_time = datetime.utcnow()
            expiry_hours = expires_in_hours or self.default_expiry_hours
            expiration_time = current_time + timedelta(hours=expiry_hours)
            
            # Validate expiration time constraints
            if expiration_time - current_time > self.max_token_lifetime:
                raise ValueError(f"Token lifetime exceeds maximum allowed: {self.max_token_lifetime}")
            
            # Generate unique token identifier
            token_id = self._generate_token_id(token_claims.subject, token_type)
            
            # Construct comprehensive JWT payload
            payload = {
                # Standard JWT claims
                "iss": token_claims.issuer,
                "sub": token_claims.subject,
                "aud": token_claims.audience,
                "exp": expiration_time,
                "iat": current_time,
                "nbf": current_time,
                "jti": token_id,
                
                # MAMA system specific claims
                "token_type": token_type.value,
                "scope": token_claims.scope or self._get_default_scope(token_type),
                "permissions": token_claims.permissions or self._get_default_permissions(token_type),
                "trust_level": token_claims.trust_level,
                
                # Agent-specific claims
                "agent_capabilities": token_claims.agent_capabilities or [],
                "security_context": token_claims.security_context or {},
                
                # Milestone API specific permissions
                "milestone_permissions": self._get_milestone_permissions(token_type),
                
                # Academic system integration
                "mama_version": "1.0.0",
                "academic_mode": True,
                "trust_aware": True,
                "multi_agent_enabled": True
            }
            
            # Add custom claims if provided
            if custom_claims:
                # Ensure custom claims don't override standard claims
                filtered_claims = {k: v for k, v in custom_claims.items() 
                                 if k not in payload}
                payload.update(filtered_claims)
            
            # Generate signed token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            # Update generation statistics
            self.token_generation_count += 1
            
            logger.info(f"JWT token generated successfully: type={token_type.value}, "
                       f"subject={token_claims.subject}, expires={expiration_time}")
            
            return token
            
        except Exception as e:
            logger.error(f"JWT token generation failed: {e}")
            raise RuntimeError(f"Token generation failed: {str(e)}")
    
    def verify_token(self, token: str, expected_audience: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Comprehensive JWT token verification with security validation
        
        Args:
            token: JWT token string to verify
            expected_audience: Expected audience claim for validation
            
        Returns:
            Decoded token payload if valid, None if verification fails
        """
        try:
            self.token_verification_count += 1
            
            # Verify token signature and claims
            options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": expected_audience is not None,
                "require": ["exp", "iat", "sub", "iss"]
            }
            
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options=options,
                audience=expected_audience,
                leeway=self.clock_skew_tolerance
            )
            
            # Additional security validations
            if not self._validate_token_payload(payload):
                self.failed_verification_count += 1
                return None
            
            # Check token type validity
            token_type = payload.get("token_type")
            if token_type and not self._is_valid_token_type(token_type):
                logger.warning(f"Invalid token type in payload: {token_type}")
                self.failed_verification_count += 1
                return None
            
            # Validate trust level constraints
            trust_level = payload.get("trust_level", 0.0)
            if trust_level < 0.0 or trust_level > 1.0:
                logger.warning(f"Invalid trust level in token: {trust_level}")
                self.failed_verification_count += 1
                return None
            
            logger.info(f"JWT token verified successfully: subject={payload.get('sub')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            self.failed_verification_count += 1
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT token is invalid: {e}")
            self.failed_verification_count += 1
            return None
        except Exception as e:
            logger.error(f"JWT token verification failed: {e}")
            self.failed_verification_count += 1
            return None
    
    def refresh_token(self, 
                     token: str, 
                     extend_hours: int = 24,
                     preserve_claims: bool = True) -> Optional[str]:
        """
        Refresh JWT token with extended expiration and updated claims
        
        Args:
            token: Existing JWT token to refresh
            extend_hours: Hours to extend token validity
            preserve_claims: Whether to preserve custom claims from original token
            
        Returns:
            New JWT token string if refresh successful, None otherwise
        """
        try:
            # Verify existing token (ignore expiration for refresh)
            options = {"verify_exp": False}
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options=options
            )
            
            # Validate that token is eligible for refresh
            if not self._can_refresh_token(payload):
                logger.warning("Token is not eligible for refresh")
                return None
            
            # Extract token claims
            token_claims = TokenClaims(
                subject=payload.get("sub"),
                audience=payload.get("aud"),
                issuer=payload.get("iss", "mama_flight_assistant"),
                scope=payload.get("scope") if preserve_claims else None,
                permissions=payload.get("permissions") if preserve_claims else None,
                agent_capabilities=payload.get("agent_capabilities") if preserve_claims else None,
                trust_level=payload.get("trust_level", 0.8),
                security_context=payload.get("security_context") if preserve_claims else None
            )
            
            # Determine token type
            token_type_str = payload.get("token_type", TokenType.ACCESS_TOKEN.value)
            token_type = TokenType(token_type_str)
            
            # Generate refreshed token
            custom_claims = {}
            if preserve_claims:
                # Preserve custom claims while filtering out standard JWT claims
                standard_claims = {"iss", "sub", "aud", "exp", "iat", "nbf", "jti"}
                custom_claims = {k: v for k, v in payload.items() 
                               if k not in standard_claims and not k.startswith("token_")}
            
            new_token = self.generate_token(
                token_claims=token_claims,
                token_type=token_type,
                expires_in_hours=extend_hours,
                custom_claims=custom_claims
            )
            
            logger.info(f"JWT token refreshed successfully: subject={token_claims.subject}")
            return new_token
            
        except Exception as e:
            logger.error(f"JWT token refresh failed: {e}")
            return None

    def generate_agent_token(self, 
                           agent_id: str,
                           agent_capabilities: List[str],
                           trust_level: float = 0.8,
                           expires_in_hours: int = 12) -> str:
        """
        Generate specialized token for MAMA agent authentication
        
        Args:
            agent_id: Unique identifier for the agent
            agent_capabilities: List of agent capabilities
            trust_level: Current trust level for the agent (0.0-1.0)
            expires_in_hours: Token expiration time
            
        Returns:
            JWT token for agent authentication
        """
        token_claims = TokenClaims(
            subject=agent_id,
            audience="mama_agent_system",
            scope=["agent:execute", "agent:communicate", "agent:coordinate"],
            permissions=self._get_agent_permissions(agent_capabilities),
            agent_capabilities=agent_capabilities,
            trust_level=trust_level,
            security_context={
                "agent_type": self._classify_agent_type(agent_id),
                "creation_timestamp": datetime.utcnow().isoformat(),
                "trust_threshold": 0.7
            }
        )
        
        return self.generate_token(
            token_claims=token_claims,
            token_type=TokenType.AGENT_TOKEN,
            expires_in_hours=expires_in_hours
        )
    
    def generate_milestone_token(self, 
                               service_name: str = "mama_system",
                               permission_level: PermissionLevel = PermissionLevel.READ_WRITE,
                               expires_in_hours: int = 24) -> str:
        """
        Generate specialized token for Milestone API access
        
        Args:
            service_name: Name of the service requesting access
            permission_level: Level of permissions required
            expires_in_hours: Token expiration time
            
        Returns:
            JWT token for Milestone API authentication
        """
        token_claims = TokenClaims(
            subject=service_name,
            audience="milestone_api",
            scope=self._get_milestone_scope(permission_level),
            permissions=self._get_milestone_api_permissions(permission_level),
            trust_level=1.0,  # System-level trust for API access
            security_context={
                "api_version": "1.0",
                "permission_level": permission_level.value,
                "rate_limit": self._get_rate_limit(permission_level)
            }
        )
        
        return self.generate_token(
            token_claims=token_claims,
            token_type=TokenType.MILESTONE_TOKEN,
            expires_in_hours=expires_in_hours
        )
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """Get comprehensive token generation and verification statistics"""
        return {
            "tokens_generated": self.token_generation_count,
            "tokens_verified": self.token_verification_count,
            "verification_failures": self.failed_verification_count,
            "success_rate": (
                (self.token_verification_count - self.failed_verification_count) / 
                max(self.token_verification_count, 1)
            ),
            "algorithm": self.algorithm,
            "default_expiry_hours": self.default_expiry_hours,
            "max_token_lifetime": str(self.max_token_lifetime)
        }
    
    # Private helper methods
    
    def _validate_token_claims(self, claims: TokenClaims) -> None:
        """Validate token claims for security compliance"""
        if not claims.subject:
            raise ValueError("Token subject is required")
        if not claims.audience:
            raise ValueError("Token audience is required")
        if claims.trust_level < 0.0 or claims.trust_level > 1.0:
            raise ValueError("Trust level must be between 0.0 and 1.0")
    
    def _validate_token_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate decoded token payload for security compliance"""
        required_fields = ["iss", "sub", "aud", "exp", "iat"]
        return all(field in payload for field in required_fields)
    
    def _is_valid_token_type(self, token_type: str) -> bool:
        """Check if token type is valid"""
        return token_type in [t.value for t in TokenType]
    
    def _can_refresh_token(self, payload: Dict[str, Any]) -> bool:
        """Check if token is eligible for refresh"""
        # Check if token was issued recently enough for refresh
        issued_at = datetime.fromtimestamp(payload.get("iat", 0))
        max_refresh_age = timedelta(days=7)
        return datetime.utcnow() - issued_at < max_refresh_age
    
    def _generate_token_id(self, subject: str, token_type: TokenType) -> str:
        """Generate unique token identifier"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        content = f"{subject}:{token_type.value}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_default_scope(self, token_type: TokenType) -> List[str]:
        """Get default scope for token type"""
        scope_map = {
            TokenType.ACCESS_TOKEN: ["read", "write"],
            TokenType.REFRESH_TOKEN: ["refresh"],
            TokenType.SERVICE_TOKEN: ["service:access"],
            TokenType.AGENT_TOKEN: ["agent:execute"],
            TokenType.MILESTONE_TOKEN: ["api:access"]
        }
        return scope_map.get(token_type, ["read"])
    
    def _get_default_permissions(self, token_type: TokenType) -> List[str]:
        """Get default permissions for token type"""
        permission_map = {
            TokenType.ACCESS_TOKEN: ["read:data", "write:data"],
            TokenType.REFRESH_TOKEN: ["refresh:token"],
            TokenType.SERVICE_TOKEN: ["access:service"],
            TokenType.AGENT_TOKEN: ["execute:tasks", "communicate:agents"],
            TokenType.MILESTONE_TOKEN: ["read:entities", "write:entities", "query:realtime"]
        }
        return permission_map.get(token_type, ["read:data"])
    
    def _get_milestone_permissions(self, token_type: TokenType) -> List[str]:
        """Get Milestone API specific permissions"""
        return [
            "read:entities",
            "write:entities",
            "query:realtime",
            "access:ngsi-ld",
            "manage:subscriptions",
            "access:temporal"
        ]
    
    def _get_agent_permissions(self, capabilities: List[str]) -> List[str]:
        """Get permissions based on agent capabilities"""
        permissions = ["agent:basic"]
        
        capability_permissions = {
            "weather_analysis": ["access:weather_data", "analyze:meteorology"],
            "safety_assessment": ["access:safety_data", "analyze:risk"],
            "flight_info": ["access:flight_data", "query:schedules"],
            "economic_analysis": ["access:pricing_data", "analyze:costs"],
            "integration": ["coordinate:agents", "synthesize:decisions"]
        }
        
        for capability in capabilities:
            if capability in capability_permissions:
                permissions.extend(capability_permissions[capability])
        
        return permissions
    
    def _classify_agent_type(self, agent_id: str) -> str:
        """Classify agent type based on agent ID"""
        if "weather" in agent_id.lower():
            return "weather_agent"
        elif "safety" in agent_id.lower():
            return "safety_agent"
        elif "flight" in agent_id.lower():
            return "flight_agent"
        elif "economic" in agent_id.lower():
            return "economic_agent"
        elif "integration" in agent_id.lower():
            return "integration_agent"
        else:
            return "generic_agent"
    
    def _get_milestone_scope(self, permission_level: PermissionLevel) -> List[str]:
        """Get Milestone API scope based on permission level"""
        scope_map = {
            PermissionLevel.READ_ONLY: ["read"],
            PermissionLevel.READ_WRITE: ["read", "write"],
            PermissionLevel.ADMIN: ["read", "write", "admin"],
            PermissionLevel.SYSTEM: ["read", "write", "admin", "system"],
            PermissionLevel.AGENT_COORDINATOR: ["read", "write", "coordinate"]
        }
        return scope_map.get(permission_level, ["read"])
    
    def _get_milestone_api_permissions(self, permission_level: PermissionLevel) -> List[str]:
        """Get Milestone API permissions based on level"""
        base_permissions = ["read:entities"]
        
        if permission_level in [PermissionLevel.READ_WRITE, PermissionLevel.ADMIN, 
                               PermissionLevel.SYSTEM, PermissionLevel.AGENT_COORDINATOR]:
            base_permissions.extend(["write:entities", "query:realtime"])
        
        if permission_level in [PermissionLevel.ADMIN, PermissionLevel.SYSTEM]:
            base_permissions.extend(["manage:subscriptions", "access:temporal", "admin:entities"])
        
        if permission_level == PermissionLevel.SYSTEM:
            base_permissions.extend(["system:configure", "system:monitor"])
        
        if permission_level == PermissionLevel.AGENT_COORDINATOR:
            base_permissions.extend(["coordinate:agents", "manage:workflow"])
        
        return base_permissions
    
    def _get_rate_limit(self, permission_level: PermissionLevel) -> int:
        """Get rate limit based on permission level"""
        rate_limits = {
            PermissionLevel.READ_ONLY: 100,
            PermissionLevel.READ_WRITE: 500,
            PermissionLevel.ADMIN: 1000,
            PermissionLevel.SYSTEM: 10000,
            PermissionLevel.AGENT_COORDINATOR: 2000
        }
        return rate_limits.get(permission_level, 100)


def generate_milestone_token(service_name: str = "mama_system",
                           permission_level: PermissionLevel = PermissionLevel.READ_WRITE,
                           expires_in_hours: int = 24) -> str:
    """
    Convenience function for generating Milestone API tokens
    
    Args:
        service_name: Name of service requesting token
        permission_level: Required permission level
        expires_in_hours: Token expiration time
        
    Returns:
        JWT token string for Milestone API access
    """
    generator = MilestoneTokenGenerator()
    return generator.generate_milestone_token(service_name, permission_level, expires_in_hours)


def create_agent_token(agent_id: str,
                      capabilities: List[str],
                      trust_level: float = 0.8) -> str:
    """
    Convenience function for generating agent authentication tokens
    
    Args:
        agent_id: Unique agent identifier
        capabilities: List of agent capabilities
        trust_level: Current trust level (0.0-1.0)
    
    Returns:
        JWT token string for agent authentication
    """
    generator = MilestoneTokenGenerator()
    return generator.generate_agent_token(agent_id, capabilities, trust_level)


# Academic integration functions

def evaluate_mama_performance(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate MAMA system performance using academic metrics
    
    Args:
        results: List of system evaluation results
        
    Returns:
        Dictionary with performance metrics
    """
    from .evaluation_metrics import calculate_mrr, calculate_ndcg, calculate_art
    
    return {
        "mean_reciprocal_rank": calculate_mrr(results),
        "ndcg_at_5": calculate_ndcg(results, k=5),
        "average_response_time": calculate_art(results)
    }


if __name__ == "__main__":
    # Generate demonstration tokens
    generator = MilestoneTokenGenerator()
    
    # Generate Milestone API token
    milestone_token = generator.generate_milestone_token()
    print(f"Milestone API Token: {milestone_token}")
    
    # Generate agent token
    agent_token = generator.generate_agent_token(
        agent_id="weather_agent_001",
        agent_capabilities=["weather_analysis", "meteorological_forecasting"],
        trust_level=0.85
    )
    print(f"Agent Token: {agent_token}")
    
    # Display statistics
    stats = generator.get_token_statistics()
    print(f"Token Statistics: {stats}")
