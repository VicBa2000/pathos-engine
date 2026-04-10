"""Simple encryption for API keys in session saves.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
The encryption key is auto-generated and stored locally in .pathos_key.
This is NOT a security boundary — it prevents casual exposure of API keys
in JSON files, not a determined attacker with filesystem access.
"""

import base64
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Key file next to saves/ directory
_KEY_FILE = Path(__file__).parent.parent.parent.parent / ".pathos_key"

# Prefix to identify encrypted values in JSON
_ENCRYPTED_PREFIX = "enc::"


def _get_or_create_key() -> bytes:
    """Gets the local encryption key, creating it if needed."""
    if _KEY_FILE.exists():
        raw = _KEY_FILE.read_text(encoding="utf-8").strip()
        # Fernet key must be 32 url-safe base64-encoded bytes
        return raw.encode("utf-8")

    # Generate a new key
    try:
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
    except ImportError:
        # Fallback: generate a compatible key without cryptography
        import secrets
        key = base64.urlsafe_b64encode(secrets.token_bytes(32))

    _KEY_FILE.write_text(key.decode("utf-8"), encoding="utf-8")
    # Restrict permissions on Unix systems (Windows ignores this)
    try:
        import os
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass
    logger.info("Generated new encryption key at %s", _KEY_FILE)
    return key


def _get_fernet():
    """Returns a Fernet instance, or None if cryptography is not installed."""
    try:
        from cryptography.fernet import Fernet
        return Fernet(_get_or_create_key())
    except ImportError:
        return None


def encrypt_value(value: str) -> str:
    """Encrypts a string value. Returns prefixed encrypted string.

    If cryptography is not installed, returns the value as-is (no encryption).
    """
    if not value:
        return value

    fernet = _get_fernet()
    if fernet is None:
        # No encryption available — return plaintext
        return value

    encrypted = fernet.encrypt(value.encode("utf-8"))
    return _ENCRYPTED_PREFIX + encrypted.decode("utf-8")


def decrypt_value(value: str) -> str:
    """Decrypts a prefixed encrypted string. Returns plaintext.

    Handles: encrypted values, plaintext values (legacy), empty values.
    """
    if not value:
        return value

    # Not encrypted (legacy save or no encryption)
    if not value.startswith(_ENCRYPTED_PREFIX):
        return value

    fernet = _get_fernet()
    if fernet is None:
        logger.warning("Cannot decrypt: cryptography not installed. Returning empty.")
        return ""

    try:
        encrypted_data = value[len(_ENCRYPTED_PREFIX):].encode("utf-8")
        return fernet.decrypt(encrypted_data).decode("utf-8")
    except Exception as e:
        logger.warning("Decryption failed (key changed?): %s", e)
        return ""


def encrypt_cloud_providers(providers: dict) -> dict:
    """Encrypts API keys in cloud_providers dict for saving."""
    encrypted = {}
    for pid, cfg in providers.items():
        cfg_copy = dict(cfg)
        if "api_key" in cfg_copy and cfg_copy["api_key"]:
            cfg_copy["api_key"] = encrypt_value(cfg_copy["api_key"])
        encrypted[pid] = cfg_copy
    return encrypted


def decrypt_cloud_providers(providers: dict) -> dict:
    """Decrypts API keys in cloud_providers dict after loading."""
    decrypted = {}
    for pid, cfg in providers.items():
        cfg_copy = dict(cfg)
        if "api_key" in cfg_copy and cfg_copy["api_key"]:
            cfg_copy["api_key"] = decrypt_value(cfg_copy["api_key"])
        decrypted[pid] = cfg_copy
    return decrypted
