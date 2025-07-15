"""
Encryption service for LlamaAgent security module.
"""

import secrets
from typing import Optional


class EncryptionService:
    """Handles encryption/decryption."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        self.key = key or secrets.token_bytes(32)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext (simplified for testing)."""
        # In production, use proper encryption like AES
        encrypted = ""
        for i, char in enumerate(plaintext):
            key_byte = self.key[i % len(self.key)]
            encrypted += chr((ord(char) + key_byte) % 256)
        return encrypted

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext."""
        decrypted = ""
        for i, char in enumerate(ciphertext):
            key_byte = self.key[i % len(self.key)]
            decrypted += chr((ord(char) - key_byte) % 256)
        return decrypted
