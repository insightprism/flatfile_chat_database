"""
FF Security Validator - Input Validation and Security Checks

Provides comprehensive input validation and security checks for
tool parameters, commands, and data to prevent security vulnerabilities.
"""

import re
import os
import hashlib
import mimetypes
import urllib.parse
from typing import Dict, Any, List, Optional, Union, Pattern, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from ff_utils.ff_logging import get_logger
from ff_class_configs.ff_tools_config import FFToolsSecurityConfigDTO

logger = get_logger(__name__)


class FFValidationSeverity(Enum):
    """Severity levels for validation results"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FFValidationCategory(Enum):
    """Categories of validation checks"""
    INPUT_SANITIZATION = "input_sanitization"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    SQL_INJECTION = "sql_injection"
    XSS_PREVENTION = "xss_prevention"
    FILE_VALIDATION = "file_validation"
    URL_VALIDATION = "url_validation"
    DATA_PRIVACY = "data_privacy"
    RESOURCE_LIMITS = "resource_limits"
    CONTENT_FILTERING = "content_filtering"


@dataclass
class FFValidationRule:
    """Individual validation rule"""
    name: str
    category: FFValidationCategory
    severity: FFValidationSeverity
    pattern: Optional[Pattern] = None
    validator_function: Optional[Callable] = None
    error_message: str = ""
    enabled: bool = True


@dataclass
class FFValidationResult:
    """Result of validation check"""
    valid: bool
    severity: FFValidationSeverity
    category: FFValidationCategory
    rule_name: str
    error_message: str = ""
    sanitized_value: Optional[Any] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class FFSecurityValidator:
    """
    Comprehensive security validator for tool inputs and commands.
    
    Provides multiple layers of validation including input sanitization,
    path traversal prevention, command injection detection, and more.
    """
    
    def __init__(self, security_config: FFToolsSecurityConfigDTO):
        """
        Initialize FF security validator.
        
        Args:
            security_config: Security configuration
        """
        self.security_config = security_config
        self.logger = get_logger(__name__)
        
        # Validation rules
        self.validation_rules: Dict[str, FFValidationRule] = {}
        
        # Compiled patterns for performance
        self.compiled_patterns: Dict[str, Pattern] = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        self._compile_patterns()
    
    def _initialize_validation_rules(self):
        """Initialize built-in validation rules"""
        
        # Path traversal prevention
        self.validation_rules["path_traversal"] = FFValidationRule(
            name="path_traversal",
            category=FFValidationCategory.PATH_TRAVERSAL,
            severity=FFValidationSeverity.CRITICAL,
            pattern=re.compile(r'\.\.[\\/]|[\\/]\.\.|\.\.[\\]|[\\]\.\.|^\.\.'),
            error_message="Path traversal attempt detected"
        )
        
        # Command injection prevention
        self.validation_rules["command_injection"] = FFValidationRule(
            name="command_injection",
            category=FFValidationCategory.COMMAND_INJECTION,
            severity=FFValidationSeverity.CRITICAL,
            pattern=re.compile(r'''[;&|`$(){}[\]<>'"\\]|&&|\|\||>>|<<'''),
            error_message="Command injection attempt detected"
        )
        
        # SQL injection prevention
        self.validation_rules["sql_injection"] = FFValidationRule(
            name="sql_injection",
            category=FFValidationCategory.SQL_INJECTION,
            severity=FFValidationSeverity.CRITICAL,
            pattern=re.compile(r'''(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)[\s\(]'''),
            error_message="SQL injection attempt detected"
        )
        
        # XSS prevention
        self.validation_rules["xss_prevention"] = FFValidationRule(
            name="xss_prevention",
            category=FFValidationCategory.XSS_PREVENTION,
            severity=FFValidationSeverity.ERROR,
            pattern=re.compile(r'<script|javascript:|vbscript:|onload=|onerror=|onclick=', re.IGNORECASE),
            error_message="Cross-site scripting attempt detected"
        )
        
        # Null byte injection
        self.validation_rules["null_byte_injection"] = FFValidationRule(
            name="null_byte_injection",
            category=FFValidationCategory.INPUT_SANITIZATION,
            severity=FFValidationSeverity.CRITICAL,
            pattern=re.compile(r'\x00'),
            error_message="Null byte injection detected"
        )
        
        # LDAP injection
        self.validation_rules["ldap_injection"] = FFValidationRule(
            name="ldap_injection",
            category=FFValidationCategory.INPUT_SANITIZATION,
            severity=FFValidationSeverity.ERROR,
            pattern=re.compile(r'[()&|!*\\]'),
            error_message="LDAP injection attempt detected"
        )
        
        # Email injection
        self.validation_rules["email_injection"] = FFValidationRule(
            name="email_injection",
            category=FFValidationCategory.INPUT_SANITIZATION,
            severity=FFValidationSeverity.ERROR,
            pattern=re.compile(r'[\r\n]+(to|cc|bcc|subject):', re.IGNORECASE),
            error_message="Email header injection detected"
        )
        
        # File extension validation
        self.validation_rules["dangerous_file_extension"] = FFValidationRule(
            name="dangerous_file_extension",
            category=FFValidationCategory.FILE_VALIDATION,
            severity=FFValidationSeverity.WARNING,
            pattern=re.compile(r'\.(exe|bat|cmd|com|scr|pif|vbs|js|jar|app|deb|rpm)$', re.IGNORECASE),
            error_message="Potentially dangerous file extension detected"
        )
        
        # URL validation
        self.validation_rules["malicious_url"] = FFValidationRule(
            name="malicious_url",
            category=FFValidationCategory.URL_VALIDATION,
            severity=FFValidationSeverity.WARNING,
            pattern=re.compile(r'(?i)(javascript|data|vbscript|file|ftp):', re.IGNORECASE),
            error_message="Potentially malicious URL scheme detected"
        )
        
        # Privacy - potential PII detection
        self.validation_rules["potential_ssn"] = FFValidationRule(
            name="potential_ssn",
            category=FFValidationCategory.DATA_PRIVACY,
            severity=FFValidationSeverity.WARNING,
            pattern=re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            error_message="Potential Social Security Number detected"
        )
        
        self.validation_rules["potential_credit_card"] = FFValidationRule(
            name="potential_credit_card",
            category=FFValidationCategory.DATA_PRIVACY,
            severity=FFValidationSeverity.WARNING,
            pattern=re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            error_message="Potential credit card number detected"
        )
        
        # Content filtering
        self.validation_rules["profanity_filter"] = FFValidationRule(
            name="profanity_filter",
            category=FFValidationCategory.CONTENT_FILTERING,
            severity=FFValidationSeverity.INFO,
            validator_function=self._check_profanity,
            error_message="Inappropriate content detected"
        )
    
    def _compile_patterns(self):
        """Pre-compile regular expressions for better performance"""
        for rule_name, rule in self.validation_rules.items():
            if rule.pattern:
                self.compiled_patterns[rule_name] = rule.pattern
    
    def validate_input(self, 
                      value: Any,
                      input_type: str = "general",
                      rules: Optional[List[str]] = None) -> List[FFValidationResult]:
        """
        Validate input value against security rules.
        
        Args:
            value: Value to validate
            input_type: Type of input (general, command, file_path, url, etc.)
            rules: Specific rules to apply (if None, applies relevant rules)
            
        Returns:
            List[FFValidationResult]: Validation results
        """
        results = []
        
        try:
            # Convert value to string for pattern matching
            str_value = str(value) if value is not None else ""
            
            # Determine which rules to apply
            applicable_rules = self._get_applicable_rules(input_type, rules)
            
            # Apply each rule
            for rule_name in applicable_rules:
                rule = self.validation_rules.get(rule_name)
                if not rule or not rule.enabled:
                    continue
                
                result = self._apply_validation_rule(str_value, rule)
                if result:
                    results.append(result)
            
            # If no violations found, add success result
            if not results:
                results.append(FFValidationResult(
                    valid=True,
                    severity=FFValidationSeverity.INFO,
                    category=FFValidationCategory.INPUT_SANITIZATION,
                    rule_name="validation_passed",
                    sanitized_value=value
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return [FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=FFValidationCategory.INPUT_SANITIZATION,
                rule_name="validation_error",
                error_message=f"Validation failed: {str(e)}"
            )]
    
    def _get_applicable_rules(self, input_type: str, custom_rules: Optional[List[str]]) -> List[str]:
        """Get list of applicable validation rules for input type"""
        if custom_rules:
            return custom_rules
        
        # Define rule sets for different input types
        rule_sets = {
            "general": [
                "null_byte_injection", "xss_prevention", "sql_injection",
                "potential_ssn", "potential_credit_card", "profanity_filter"
            ],
            "command": [
                "null_byte_injection", "command_injection", "path_traversal"
            ],
            "file_path": [
                "null_byte_injection", "path_traversal", "dangerous_file_extension"
            ],
            "url": [
                "null_byte_injection", "malicious_url", "xss_prevention"
            ],
            "email": [
                "null_byte_injection", "email_injection", "xss_prevention"
            ],
            "database": [
                "null_byte_injection", "sql_injection", "ldap_injection"
            ],
            "content": [
                "xss_prevention", "profanity_filter", "potential_ssn", "potential_credit_card"
            ]
        }
        
        return rule_sets.get(input_type, rule_sets["general"])
    
    def _apply_validation_rule(self, value: str, rule: FFValidationRule) -> Optional[FFValidationResult]:
        """Apply a single validation rule to a value"""
        try:
            violation_found = False
            
            # Apply pattern-based validation
            if rule.pattern and rule.pattern.search(value):
                violation_found = True
            
            # Apply function-based validation
            if rule.validator_function:
                if rule.validator_function(value):
                    violation_found = True
            
            # Return result if violation found
            if violation_found:
                return FFValidationResult(
                    valid=False,
                    severity=rule.severity,
                    category=rule.category,
                    rule_name=rule.name,
                    error_message=rule.error_message,
                    sanitized_value=self._sanitize_value(value, rule)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.name}: {e}")
            return FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=rule.category,
                rule_name=rule.name,
                error_message=f"Rule application failed: {str(e)}"
            )
    
    def _sanitize_value(self, value: str, rule: FFValidationRule) -> str:
        """Sanitize value based on the validation rule"""
        try:
            if rule.category == FFValidationCategory.XSS_PREVENTION:
                # Basic HTML escaping
                return (value.replace('<', '&lt;')
                           .replace('>', '&gt;')
                           .replace('"', '&quot;')
                           .replace("'", '&#x27;'))
            
            elif rule.category == FFValidationCategory.PATH_TRAVERSAL:
                # Remove path traversal sequences
                return re.sub(r'\.\.[\\/]|[\\/]\.\.|\.\.[\\]|[\\]\.\.|^\.\.', '', value)
            
            elif rule.category == FFValidationCategory.COMMAND_INJECTION:
                # Remove dangerous characters
                return re.sub(r'''[;&|`$(){}[\]<>'"\\]''', '', value)
            
            elif rule.category == FFValidationCategory.SQL_INJECTION:
                # Escape SQL dangerous keywords
                return re.sub(r'''(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)''', 
                            lambda m: f"[{m.group(1)}]", value)
            
            elif rule.category == FFValidationCategory.DATA_PRIVACY:
                # Mask sensitive data
                if "ssn" in rule.name:
                    return re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', 'XXX-XX-XXXX', value)
                elif "credit_card" in rule.name:
                    return re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', value)
            
            # Default: return original value
            return value
            
        except Exception as e:
            self.logger.error(f"Sanitization error: {e}")
            return value
    
    def _check_profanity(self, value: str) -> bool:
        """Check for profanity/inappropriate content (simplified implementation)"""
        # This is a simplified implementation - in production, use a comprehensive profanity filter
        profane_words = [
            'damn', 'hell', 'crap', 'shit', 'fuck', 'bitch', 'asshole'
        ]
        
        value_lower = value.lower()
        return any(word in value_lower for word in profane_words)
    
    def validate_file_path(self, file_path: str) -> FFValidationResult:
        """Validate file path for security issues"""
        try:
            path = Path(file_path)
            
            # Check for path traversal
            if '..' in file_path:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.CRITICAL,
                    category=FFValidationCategory.PATH_TRAVERSAL,
                    rule_name="path_traversal",
                    error_message="Path traversal detected in file path"
                )
            
            # Check against blocked paths
            for blocked_path in self.security_config.blocked_file_paths:
                if file_path.startswith(blocked_path):
                    return FFValidationResult(
                        valid=False,
                        severity=FFValidationSeverity.ERROR,
                        category=FFValidationCategory.FILE_VALIDATION,
                        rule_name="blocked_path",
                        error_message=f"Access to blocked path: {blocked_path}"
                    )
            
            # Check file extension
            file_extension = path.suffix.lower()
            if file_extension not in self.security_config.allowed_file_extensions:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.WARNING,
                    category=FFValidationCategory.FILE_VALIDATION,
                    rule_name="file_extension",
                    error_message=f"File extension {file_extension} not allowed"
                )
            
            return FFValidationResult(
                valid=True,
                severity=FFValidationSeverity.INFO,
                category=FFValidationCategory.FILE_VALIDATION,
                rule_name="file_path_valid"
            )
            
        except Exception as e:
            return FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=FFValidationCategory.FILE_VALIDATION,
                rule_name="validation_error",
                error_message=f"File path validation failed: {str(e)}"
            )
    
    def validate_url(self, url: str) -> FFValidationResult:
        """Validate URL for security issues"""
        try:
            parsed_url = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed_url.scheme not in ['http', 'https']:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.ERROR,
                    category=FFValidationCategory.URL_VALIDATION,
                    rule_name="invalid_scheme",
                    error_message=f"URL scheme {parsed_url.scheme} not allowed"
                )
            
            # Check against blocked domains
            if parsed_url.netloc in self.security_config.blocked_domains:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.ERROR,
                    category=FFValidationCategory.URL_VALIDATION,
                    rule_name="blocked_domain",
                    error_message=f"Domain {parsed_url.netloc} is blocked"
                )
            
            # Check against allowed domains (if specified)
            if self.security_config.allowed_domains:
                if parsed_url.netloc not in self.security_config.allowed_domains:
                    return FFValidationResult(
                        valid=False,
                        severity=FFValidationSeverity.ERROR,
                        category=FFValidationCategory.URL_VALIDATION,
                        rule_name="domain_not_allowed",
                        error_message=f"Domain {parsed_url.netloc} not in allowed list"
                    )
            
            return FFValidationResult(
                valid=True,
                severity=FFValidationSeverity.INFO,
                category=FFValidationCategory.URL_VALIDATION,
                rule_name="url_valid"
            )
            
        except Exception as e:
            return FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=FFValidationCategory.URL_VALIDATION,
                rule_name="validation_error",
                error_message=f"URL validation failed: {str(e)}"
            )
    
    def validate_command(self, command: str, args: List[str] = None) -> FFValidationResult:
        """Validate command and arguments for security issues"""
        try:
            # Extract base command
            base_command = command.split()[0] if ' ' in command else command
            base_command = os.path.basename(base_command)
            
            # Check against blocked commands
            if base_command in self.security_config.blocked_commands:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.CRITICAL,
                    category=FFValidationCategory.COMMAND_INJECTION,
                    rule_name="blocked_command",
                    error_message=f"Command {base_command} is blocked"
                )
            
            # Check against allowed commands
            if self.security_config.allowed_commands:
                if base_command not in self.security_config.allowed_commands:
                    return FFValidationResult(
                        valid=False,
                        severity=FFValidationSeverity.ERROR,
                        category=FFValidationCategory.COMMAND_INJECTION,
                        rule_name="command_not_allowed",
                        error_message=f"Command {base_command} not in allowed list"
                    )
            
            # Validate full command for injection
            full_command = command + (' ' + ' '.join(args) if args else '')
            injection_results = self.validate_input(full_command, "command")
            
            # Check for critical security issues
            for result in injection_results:
                if not result.valid and result.severity == FFValidationSeverity.CRITICAL:
                    return result
            
            return FFValidationResult(
                valid=True,
                severity=FFValidationSeverity.INFO,
                category=FFValidationCategory.COMMAND_INJECTION,
                rule_name="command_valid"
            )
            
        except Exception as e:
            return FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=FFValidationCategory.COMMAND_INJECTION,
                rule_name="validation_error",
                error_message=f"Command validation failed: {str(e)}"
            )
    
    def validate_file_content(self, file_path: str, max_size_mb: int = 10) -> FFValidationResult:
        """Validate file content for security issues"""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.ERROR,
                    category=FFValidationCategory.FILE_VALIDATION,
                    rule_name="file_not_found",
                    error_message="File does not exist"
                )
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.ERROR,
                    category=FFValidationCategory.RESOURCE_LIMITS,
                    rule_name="file_too_large",
                    error_message=f"File size {file_size_mb:.2f}MB exceeds limit {max_size_mb}MB"
                )
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            dangerous_mime_types = [
                'application/x-executable',
                'application/x-msdownload',
                'application/x-sh',
                'text/x-shellscript'
            ]
            
            if mime_type in dangerous_mime_types:
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.WARNING,
                    category=FFValidationCategory.FILE_VALIDATION,
                    rule_name="dangerous_mime_type",
                    error_message=f"Potentially dangerous MIME type: {mime_type}"
                )
            
            # Basic content scan for executable signatures
            with open(path, 'rb') as f:
                header = f.read(4)
                
            # Check for executable headers
            executable_headers = [
                b'\x7fELF',  # ELF
                b'MZ',       # PE
                b'\xfe\xed\xfa',  # Mach-O
            ]
            
            if any(header.startswith(sig) for sig in executable_headers):
                return FFValidationResult(
                    valid=False,
                    severity=FFValidationSeverity.WARNING,
                    category=FFValidationCategory.FILE_VALIDATION,
                    rule_name="executable_file",
                    error_message="File appears to be executable"
                )
            
            return FFValidationResult(
                valid=True,
                severity=FFValidationSeverity.INFO,
                category=FFValidationCategory.FILE_VALIDATION,
                rule_name="file_content_valid"
            )
            
        except Exception as e:
            return FFValidationResult(
                valid=False,
                severity=FFValidationSeverity.ERROR,
                category=FFValidationCategory.FILE_VALIDATION,
                rule_name="validation_error",
                error_message=f"File content validation failed: {str(e)}"
            )
    
    def add_custom_rule(self, rule: FFValidationRule) -> bool:
        """Add a custom validation rule"""
        try:
            self.validation_rules[rule.name] = rule
            if rule.pattern:
                self.compiled_patterns[rule.name] = rule.pattern
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")
            return False
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule"""
        try:
            if rule_name in self.validation_rules:
                del self.validation_rules[rule_name]
            if rule_name in self.compiled_patterns:
                del self.compiled_patterns[rule_name]
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove rule: {e}")
            return False
    
    def get_validation_summary(self, results: List[FFValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        summary = {
            "total_checks": len(results),
            "passed": 0,
            "warnings": 0,
            "errors": 0,
            "critical": 0,
            "overall_valid": True,
            "categories": {},
            "highest_severity": FFValidationSeverity.INFO
        }
        
        for result in results:
            # Count by severity
            if result.severity == FFValidationSeverity.INFO:
                if result.valid:
                    summary["passed"] += 1
            elif result.severity == FFValidationSeverity.WARNING:
                summary["warnings"] += 1
                if result.severity.value > summary["highest_severity"].value:
                    summary["highest_severity"] = result.severity
            elif result.severity == FFValidationSeverity.ERROR:
                summary["errors"] += 1
                summary["overall_valid"] = False
                if result.severity.value > summary["highest_severity"].value:
                    summary["highest_severity"] = result.severity
            elif result.severity == FFValidationSeverity.CRITICAL:
                summary["critical"] += 1
                summary["overall_valid"] = False
                summary["highest_severity"] = result.severity
            
            # Count by category
            category = result.category.value
            if category not in summary["categories"]:
                summary["categories"][category] = 0
            summary["categories"][category] += 1
        
        return summary