"""
RAD50 encoding/decoding for ACNET task names.

RAD50 packs 6 characters into a 32-bit integer using base-40 encoding.
Each group of 3 characters maps to a 16-bit value (40^3 = 64000).
"""

# Character set: 40 characters (space, A-Z, $, ., %, 0-9)
RAD50_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ$.%0123456789"


def _char_to_index(c: str) -> int:
    """Convert a character to its RAD50 index (0-39)."""
    c = c.upper()
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 1
    elif "0" <= c <= "9":
        return ord(c) - ord("0") + 30
    elif c == "$":
        return 27
    elif c == ".":
        return 28
    elif c == "%":
        return 29
    return 0  # Space or invalid


def encode(s: str) -> int:
    """
    Encode a 6-character string to a 32-bit RAD50 integer.

    Args:
        s: String to encode (1-6 characters, padded with spaces if shorter)

    Returns:
        32-bit integer encoding of the string

    Example:
        >>> encode("DPM")
        0x000004A2
    """
    # Pad or truncate to exactly 6 characters
    s = s.ljust(6)[:6]

    # Encode first 3 characters -> low 16 bits
    v1 = 0
    for i in range(3):
        v1 = v1 * 40 + _char_to_index(s[i])

    # Encode last 3 characters -> high 16 bits
    v2 = 0
    for i in range(3, 6):
        v2 = v2 * 40 + _char_to_index(s[i])

    return (v2 << 16) | v1


def decode(rad50: int) -> str:
    """
    Decode a 32-bit RAD50 integer to a 6-character string.

    Args:
        rad50: 32-bit RAD50 encoded value

    Returns:
        6-character string (may have trailing spaces)

    Example:
        >>> decode(0x000004A2)
        'DPM   '
    """
    v1 = rad50 & 0xFFFF
    v2 = (rad50 >> 16) & 0xFFFF

    chars = []

    # Decode low 16 bits -> first 3 characters (in reverse order)
    for _ in range(3):
        chars.append(RAD50_CHARS[v1 % 40])
        v1 //= 40

    # Reverse to get correct order
    chars = chars[::-1]

    # Decode high 16 bits -> last 3 characters
    last3 = []
    for _ in range(3):
        last3.append(RAD50_CHARS[v2 % 40])
        v2 //= 40

    chars.extend(last3[::-1])

    return "".join(chars)


def decode_stripped(rad50: int) -> str:
    """
    Decode RAD50 and strip trailing spaces.

    Args:
        rad50: 32-bit RAD50 encoded value

    Returns:
        Decoded string with trailing spaces removed
    """
    return decode(rad50).rstrip()
