class GPUDisabledException(Exception):
    """Raised when a GPU operation is attempted while GPU computation is disabled.

    This exception is thrown when a function or method tries to perform
    a GPU-based operation, but GPU computation has been explicitly disabled
    or is unavailable in the current environment.

    Attributes:
        None

    Example:
        >>> try:
        ...     perform_gpu_operation()
        ... except GPUDisabledException:
        ...     print("GPU operations are currently disabled.")
    """
