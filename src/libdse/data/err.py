class EntryPointError(Exception):
    """Raised when the dataset entry point is not a valid root.

    For LibriSpeech, the expected layout is a directory containing
    exactly one child named ``LibriSpeech/``.  Any deviation indicates
    a wrong path or a manually altered dataset.

    For DEMAND, the expected layout is a directory containing one or more
    child directories named after the requested noise types.  Any deviation
    indicates a wrong path or a manually altered dataset.
    """

    pass
