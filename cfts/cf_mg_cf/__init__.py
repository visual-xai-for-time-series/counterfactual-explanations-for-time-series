from .mg_cf import (
    mg_cf_generate,
    mg_cf_explain,
    mg_cf_batch,
    detach_to_numpy,
    numpy_to_torch
)

# STUMPY-based optimized versions
try:
    from .mg_cf_stumpy import (
        mine_motifs_stumpy,
        mg_cf_generate_stumpy,
        mg_cf_batch_stumpy
    )
    _stumpy_available = True
except ImportError:
    _stumpy_available = False
    mine_motifs_stumpy = None
    mg_cf_generate_stumpy = None
    mg_cf_batch_stumpy = None

__all__ = [
    'mine_motifs',
    'mg_cf_generate', 
    'mg_cf_batch',
    'detach_to_numpy',
    'numpy_to_torch',
    # STUMPY versions
    'mine_motifs_stumpy',
    'mg_cf_generate_stumpy',
    'mg_cf_batch_stumpy'
]
