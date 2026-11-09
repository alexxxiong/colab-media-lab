"""Load the matching pinned FP4 Python API and native ABI before FastVideo imports."""
import json
from pathlib import Path
import sys
from study_runtime import event,sha

def apply():
    overlay=Path('/content/colab-lab/h3-kernel-overlay')
    receipt=json.loads((overlay/'ready.json').read_text())
    assert receipt['source_revision']=='ce970eb715215c79a169658d7aa6c8a96e64891e'
    assert receipt['architecture']=='sm_120a'
    for name,record in receipt['files'].items():
        assert sha(overlay/name)==record['sha256'], 'Native FP4 overlay checksum mismatch: '+name
    sys.path.insert(0,str(overlay))
    import fp4attn_cuda
    import attn_qat_infer.api as api
    assert hasattr(fp4attn_cuda,'fwd_sparse') and hasattr(api,'vsa_tile_mask_to_fp4_blocks')
    event('pinned_fp4_kernel_overlay_verified',source_revision=receipt['source_revision'])

