import os
import re

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'json'}


def allowed_file(filename):
    """
    Check if a filename has an allowed extension

    Args:
        filename (str): The filename to check

    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(file_path):
    """
    Get the file extension of a file

    Args:
        file_path (str): Path to the file

    Returns:
        str: File extension including the dot
    """
    return os.path.splitext(file_path)[1].lower()


def parse_file_patterns(patterns_str):
    """
    Parse a comma-separated string of file patterns into a list

    Args:
        patterns_str (str): Comma-separated string of file patterns (e.g., "*.txt, *.csv")

    Returns:
        list: List of file pattern strings
    """
    if not patterns_str:
        return []
    return [pattern.strip() for pattern in patterns_str.split(',') if pattern.strip()]


def pattern_matches(filename, patterns):
    """
    Check if a filename matches any of the patterns

    Args:
        filename (str): The filename to check
        patterns (list): List of file patterns (e.g., ["*.txt", "*.csv"])

    Returns:
        bool: True if the filename matches any pattern, False otherwise
    """
    if not patterns:
        return True  # If no patterns provided, match everything

    for pattern in patterns:
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('.', '\\.').replace('*', '.*')
        if re.match(f"^{regex_pattern}$", filename, re.IGNORECASE):
            return True

    return False