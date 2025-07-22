"""
Input Validator for LlamaAgent.

Provides input validation and sanitization functionality.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class SecurityLevel(Enum):
    """Security levels for validation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRICT = "strict"


@dataclass
class ValidationRule:
    """Configuration for a validation rule."""

    name: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = False
    sanitizer: Optional[Callable[[Any], Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternValidator:
    """Pattern-based validation utilities."""

    # Common regex patterns
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    URL_PATTERN = re.compile(
        r"^https?://(?:[-\w.])+(?::[0-9]+)?(?:/[^?\s]*)?(?:\?[^#\s]*)?(?:#[^\s]*)?$"
    )
    IPV4_PATTERN = re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )
    IPV6_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$")
    ALPHANUMERIC_PATTERN = re.compile(r"^[a-zA-Z0-9]+$")
    USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")

    # Security patterns
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        re.IGNORECASE,
    )
    XSS_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
    COMMAND_INJECTION_PATTERN = re.compile(r"[;&|`$(){}\[\]<>]")

    @classmethod
    def validate_email(cls, value: str) -> bool:
        """Validate email address."""
        return bool(cls.EMAIL_PATTERN.match(str(value)))

    @classmethod
    def validate_url(cls, value: str) -> bool:
        """Validate URL."""
        return bool(cls.URL_PATTERN.match(str(value)))

    @classmethod
    def validate_ip(cls, value: str) -> bool:
        """Validate IP address (v4 or v6)."""
        str_value = str(value)
        return bool(
            cls.IPV4_PATTERN.match(str_value) or cls.IPV6_PATTERN.match(str_value)
        )

    @classmethod
    def validate_alphanumeric(cls, value: str) -> bool:
        """Validate alphanumeric string."""
        return bool(cls.ALPHANUMERIC_PATTERN.match(str(value)))

    @classmethod
    def validate_username(cls, value: str) -> bool:
        """Validate username format."""
        return bool(cls.USERNAME_PATTERN.match(str(value)))

    @classmethod
    def check_sql_injection(cls, value: str) -> bool:
        """Check for SQL injection patterns."""
        return not bool(cls.SQL_INJECTION_PATTERN.search(str(value)))

    @classmethod
    def check_xss(cls, value: str) -> bool:
        """Check for XSS patterns."""
        return not bool(cls.XSS_PATTERN.search(str(value)))

    @classmethod
    def check_command_injection(cls, value: str) -> bool:
        """Check for command injection patterns."""
        return not bool(cls.COMMAND_INJECTION_PATTERN.search(str(value)))


class Sanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_html(value: str) -> str:
        """Sanitize HTML content."""
        return html.escape(str(value))

    @staticmethod
    def sanitize_sql(value: str) -> str:
        """Basic SQL sanitization."""
        str_value = str(value)
        # Replace single quotes with double quotes
        str_value = str_value.replace("'", "''")
        # Remove or escape dangerous characters
        str_value = re.sub(r"[;&|`$(){}\[\]<>]", "", str_value)
        return str_value

    @staticmethod
    def sanitize_filename(value: str) -> str:
        """Sanitize filename."""
        str_value = str(value)
        # Remove path traversal attempts
        str_value = str_value.replace("..", "")
        str_value = str_value.replace("/", "")
        str_value = str_value.replace("\\", "")
        # Remove or replace invalid characters
        str_value = re.sub(r'[<>:"|?*]', "_", str_value)
        return str_value[:255]  # Limit length

    @staticmethod
    def sanitize_url(value: str) -> str:
        """Sanitize URL."""
        str_value = str(value)
        try:
            parsed = urllib.parse.urlparse(str_value)
            if parsed.scheme not in ["http", "https"]:
                return ""
            return urllib.parse.urlunparse(parsed)
        except Exception:
            return ""

    @staticmethod
    def normalize_whitespace(value: str) -> str:
        """Normalize whitespace in string."""
        return re.sub(r"\s+", " ", str(value).strip())

    @staticmethod
    def remove_control_chars(value: str) -> str:
        """Remove control characters."""
        return "".join(
            char for char in str(value) if ord(char) >= 32 or char in "\t\n\r"
        )


class DataTypeValidator:
    """Data type validation utilities."""

    @staticmethod
    def validate_string(
        value: Any, min_length: int = 0, max_length: int = 10000
    ) -> bool:
        """Validate string type and length."""
        if not isinstance(value, str):
            return False
        return min_length <= len(value) <= max_length

    @staticmethod
    def validate_integer(
        value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None
    ) -> bool:
        """Validate integer type and range."""
        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                return False
            if max_val is not None and int_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_float(
        value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> bool:
        """Validate float type and range."""
        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                return False
            if max_val is not None and float_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """Validate boolean type."""
        return isinstance(value, bool) or str(value).lower() in (
            "true",
            "false",
            "1",
            "0",
        )

    @staticmethod
    def validate_list(
        value: Any,
        min_items: int = 0,
        max_items: int = 1000,
        item_validator: Optional[Callable[[Any], bool]] = None,
    ) -> bool:
        """Validate list type and constraints."""
        if not isinstance(value, list):
            return False

        value = cast(List[Any], value)
        if not (min_items <= len(value) <= max_items):
            return False

        if item_validator:
            return all(item_validator(item) for item in value)

        return True

    @staticmethod
    def validate_dict(
        value: Any,
        required_keys: Optional[List[str]] = None,
        key_validator: Optional[Callable[[Any], bool]] = None,
        value_validator: Optional[Callable[[Any], bool]] = None,
    ) -> bool:
        """Validate dictionary type and constraints."""
        if not isinstance(value, dict):
            return False

        value = cast(Dict[Any, Any], value)
        if required_keys:
            if not all(key in value for key in required_keys):
                return False

        if key_validator:
            if not all(key_validator(key) for key in value.keys()):
                return False

        if value_validator:
            if not all(value_validator(val) for val in value.values()):
                return False

        return True


class InputValidator:
    """Validates and sanitizes user inputs."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level

        # Common threat patterns
        self.sql_injection_patterns = [
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(?i)(exec\s*\(|sp_|xp_)",
            r"(?i)(information_schema|sysobjects|syscolumns)",
            r"(?i)(\'|\"|\-\-|\/\*|\*\/)",
        ]

        self.xss_patterns = [
            r"(?i)<script.*?>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe.*?>",
            r"(?i)<object.*?>",
            r"(?i)<embed.*?>",
        ]

        self.command_injection_patterns = [
            r"(?i)(;|\||\&|\$\(|\`)",
            r"(?i)(rm\s|wget\s|curl\s|nc\s)",
            r"(?i)(\/bin\/|\/usr\/bin\/)",
        ]

        self.prompt_injection_patterns = [
            r"(?i)(ignore\s+previous\s+instructions)",
            r"(?i)(act\s+as\s+if\s+you\s+are)",
            r"(?i)(pretend\s+to\s+be)",
            r"(?i)(disregard\s+your\s+training)",
            r"(?i)(new\s+instructions)",
        ]

    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate text input for security threats."""
        if not isinstance(text, str):
            return {
                "is_valid": False,
                "threats": ["invalid_type"],
                "sanitized": "",
                "message": "Input must be a string",
            }

        threats = []

        # Check length limits based on security level
        max_length = self._get_max_length()
        if len(text) > max_length:
            threats.append("length_exceeded")

        # Check for SQL injection
        if self._check_patterns(text, self.sql_injection_patterns):
            threats.append("sql_injection")

        # Check for XSS
        if self._check_patterns(text, self.xss_patterns):
            threats.append("xss")

        # Check for command injection
        if self._check_patterns(text, self.command_injection_patterns):
            threats.append("command_injection")

        # Check for prompt injection (specific to LLM inputs)
        if self._check_patterns(text, self.prompt_injection_patterns):
            threats.append("prompt_injection")

        # Additional checks based on security level
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.STRICT]:
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r"[^\w\s]", text)) / max(len(text), 1)
            if special_char_ratio > 0.3:
                threats.append("excessive_special_chars")

            # Check for base64 encoded content (potential obfuscation)
            if re.search(r"[A-Za-z0-9+/]{20,}={0,2}", text):
                threats.append("potential_encoded_content")

        # Sanitize the input
        sanitized = self._sanitize_text(text)

        is_valid = len(threats) == 0

        if not is_valid:
            logger.warning(f"Input validation failed: {threats}")

        return {
            "is_valid": is_valid,
            "threats": threats,
            "sanitized": sanitized,
            "message": (
                "Input is valid"
                if is_valid
                else f"Threats detected: {', '.join(threats)}"
            ),
        }

    def validate_filename(self, filename: str) -> Dict[str, Any]:
        """Validate filename for security threats."""
        if not isinstance(filename, str):
            return {
                "is_valid": False,
                "threats": ["invalid_type"],
                "sanitized": "",
                "message": "Filename must be a string",
            }

        threats = []

        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            threats.append("path_traversal")

        # Check for dangerous characters
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
        if any(char in filename for char in dangerous_chars):
            threats.append("dangerous_characters")

        # Check for system files
        system_files = ["con", "prn", "aux", "nul", "com1", "com2", "lpt1", "lpt2"]
        if filename.lower() in system_files:
            threats.append("system_filename")

        # Sanitize filename
        sanitized = re.sub(r"[^\w\-_\.]", "_", filename)

        is_valid = len(threats) == 0

        return {
            "is_valid": is_valid,
            "threats": threats,
            "sanitized": sanitized,
            "message": (
                "Filename is valid"
                if is_valid
                else f"Threats detected: {', '.join(threats)}"
            ),
        }

    def validate_url(self, url: str) -> Dict[str, Any]:
        """Validate URL for security threats."""
        if not isinstance(url, str):
            return {
                "is_valid": False,
                "threats": ["invalid_type"],
                "sanitized": "",
                "message": "URL must be a string",
            }

        threats = []

        # Check for dangerous protocols
        dangerous_protocols = ["javascript:", "data:", "file:", "ftp:"]
        if any(url.lower().startswith(proto) for proto in dangerous_protocols):
            threats.append("dangerous_protocol")

        # Check for localhost/internal IPs (in strict mode)
        if self.security_level == SecurityLevel.STRICT:
            internal_patterns = [
                r"localhost",
                r"127\.0\.0\.1",
                r"192\.168\.",
                r"10\.",
                r"172\.(1[6-9]|2[0-9]|3[0-1])\.",
            ]
            if any(re.search(pattern, url.lower()) for pattern in internal_patterns):
                threats.append("internal_url")

        # Basic URL format validation
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, url, re.IGNORECASE):
            threats.append("invalid_format")

        is_valid = len(threats) == 0

        return {
            "is_valid": is_valid,
            "threats": threats,
            "sanitized": url,  # URLs should not be modified
            "message": (
                "URL is valid"
                if is_valid
                else f"Threats detected: {', '.join(threats)}"
            ),
        }

    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the threat patterns."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _get_max_length(self) -> int:
        """Get maximum allowed text length based on security level."""
        limits = {
            SecurityLevel.LOW: 50000,
            SecurityLevel.MEDIUM: 20000,
            SecurityLevel.HIGH: 10000,
            SecurityLevel.STRICT: 5000,
        }
        return limits.get(self.security_level, 20000)

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing or escaping dangerous content."""
        # Remove or escape HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove JavaScript
        text = re.sub(r"(?i)javascript:", "", text)

        # Remove common SQL injection patterns
        text = re.sub(r"(?i)(union\s+select|drop\s+table|delete\s+from)", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Limit length
        max_length = self._get_max_length()
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    def validate_data(
        self, data: Dict[str, Any], sanitize: bool = True
    ) -> Dict[str, Any]:
        """Validate and optionally sanitize data dictionary."""
        validated_data = {}

        for key, value in data.items():
            # Validate key
            if not isinstance(key, str):
                raise ValidationError(f"Invalid key type: {type(key)}", field=key)

            # Sanitize key if needed
            clean_key = self._sanitize_text(key) if sanitize else key

            # Validate and sanitize value based on type
            if isinstance(value, str):
                validation_result = self.validate_text_input(value)
                if not validation_result["is_valid"]:
                    raise ValidationError(
                        f"Invalid text value: {validation_result['errors']}",
                        field=key,
                        value=value,
                    )
                validated_data[clean_key] = (
                    validation_result.get("sanitized", value) if sanitize else value
                )

            elif isinstance(value, (int, float)):
                # Validate numeric values
                if not DataTypeValidator.validate_integer(
                    value
                ) and not DataTypeValidator.validate_float(value):
                    raise ValidationError(
                        f"Invalid numeric value: {value}", field=key, value=value
                    )
                validated_data[clean_key] = value

            elif isinstance(value, bool):
                validated_data[clean_key] = value

            elif isinstance(value, (list, dict)):
                # For complex types, just copy for now
                validated_data[clean_key] = value

            else:
                # Convert other types to string and validate
                str_value = str(value)
                validation_result = self.validate_text_input(str_value)
                if not validation_result["is_valid"]:
                    raise ValidationError(
                        f"Invalid value: {validation_result['errors']}",
                        field=key,
                        value=value,
                    )
                validated_data[clean_key] = (
                    validation_result.get("sanitized", str_value)
                    if sanitize
                    else str_value
                )

        return validated_data

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "security_level": self.security_level.value,
            "max_text_length": self._get_max_length(),
            "patterns_count": {
                "sql_injection": len(self.sql_injection_patterns),
                "xss": len(self.xss_patterns),
                "command_injection": len(self.command_injection_patterns),
                "prompt_injection": len(self.prompt_injection_patterns),
            },
        }


class PromptValidator(InputValidator):
    """Specialized validator for LLM prompts and AI inputs."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        super().__init__(security_level)
        # Skip the broken _setup_prompt_rules() call

        # Prevent prompt injection attempts
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.STRICT]:
            self.add_rule(
                "prompt",
                ValidationRule(
                    name="prompt_injection_check",
                    validator=self._check_prompt_injection,
                    error_message="Prompt contains potential injection patterns",
                ),
            )

            self.add_rule(
                "prompt",
                ValidationRule(
                    name="system_prompt_override_check",
                    validator=self._check_system_override,
                    error_message="Prompt attempts to override system instructions",
                ),
            )

    def _check_prompt_injection(self, value: str) -> bool:
        """Check for prompt injection patterns."""
        str_value = str(value).lower()

        # Common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "ignore the above",
            "forget everything",
            "new instructions:",
            "system: ",
            "assistant: ",
            "you are now",
            "pretend to be",
            "roleplay as",
            "act as if",
            "override",
            "jailbreak",
        ]

        return not any(pattern in str_value for pattern in injection_patterns)

    def _check_system_override(self, value: str) -> bool:
        """Check for attempts to override system prompts."""
        str_value = str(value).lower()

        override_patterns = [
            "you are not",
            "do not follow",
            "ignore your",
            "your purpose is",
            "you must",
            "override previous",
            "new role:",
            "system role:",
        ]

        return not any(pattern in str_value for pattern in override_patterns)

    def validate_prompt(self, prompt: str, **context: Any) -> str:
        """Validate and sanitize an LLM prompt."""
        try:
            return self.validate_field("prompt", prompt, sanitize=True)
        except ValidationError as e:
            logger.warning(f"Prompt validation failed: {e}")
            # For prompts, we might want to clean rather than reject
            return Sanitizer.normalize_whitespace(
                Sanitizer.remove_control_chars(prompt)
            )


# Predefined validators for common use cases
def create_api_validator() -> InputValidator:
    """Create validator for API inputs."""
    validator = InputValidator(SecurityLevel.HIGH)

    # Common API field validations
    validator.add_rule(
        "email",
        ValidationRule(
            name="email_format",
            validator=PatternValidator.validate_email,
            error_message="Invalid email format",
        ),
    )

    validator.add_rule(
        "url",
        ValidationRule(
            name="url_format",
            validator=PatternValidator.validate_url,
            error_message="Invalid URL format",
            sanitizer=Sanitizer.sanitize_url,
        ),
    )

    validator.add_rule(
        "username",
        ValidationRule(
            name="username_format",
            validator=PatternValidator.validate_username,
            error_message="Username must be 3-50 characters, alphanumeric plus _ and -",
        ),
    )

    return validator


def create_file_validator() -> InputValidator:
    """Create validator for file operations."""
    validator = InputValidator(SecurityLevel.STRICT)

    validator.add_rule(
        "filename",
        ValidationRule(
            name="filename_safety",
            validator=lambda x: ".." not in str(x)
            and "/" not in str(x)
            and "\\" not in str(x),
            error_message="Filename contains unsafe path characters",
            sanitizer=Sanitizer.sanitize_filename,
        ),
    )

    return validator


__all__ = [
    "InputValidator",
    "PromptValidator",
    "ValidationError",
    "ValidationRule",
    "SecurityLevel",
    "PatternValidator",
    "Sanitizer",
    "DataTypeValidator",
    "create_api_validator",
    "create_file_validator",
]
