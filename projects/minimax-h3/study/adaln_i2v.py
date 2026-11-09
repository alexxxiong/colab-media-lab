"""Extend fixed-ladder modulation using the checkpoint's exact original projections.

This supplies missing host weights to FastVideo's unchanged native cache-miss
implementation. It never approximates or interpolates modulation values.
Formal runs require a completed, checksummed I2V table and disable this builder.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

from study_runtime import DRIVE,ROOT,event,sha,write_json

REVISION='9d0d0d3ae69365aded83fab5562f00f9f06c12aa'
SOURCE=DRIVE/'models/base/MiniMax-H3/fasth3-v2-nvfp4'/REVISION
DESTINATION=SOURCE/'_COLAB_LAB_ADALN_I2V_VIEW'

def install():
    import torch
    from safetensors import safe_open
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3AdaLayerNormModulation as Modulation
    index=json.loads((SOURCE/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    constructor=Modulation.__init__
    signature=inspect.signature(constructor)
    def init(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs)
        prefix=bound.arguments.get('prefix','')
        match=re.search(r'transformer_blocks\.(\d+)\.',prefix)
        if match: self._study_block_index=int(match.group(1))
        constructor(self,*args,**kwargs)
    Modulation.__init__=init
    forward=Modulation.forward
    def load_original(self):
        if getattr(self,'_host_weight',None) is None:
            block=self._study_block_index
            prefix=f'transformer_blocks.{block}.adaln_proj.linear.'
            started=time.monotonic()
            event('i2v_adaln_original_projection_loading',block=block)
            values={}
            for kind in ('weight','bias'):
                name=prefix+kind
                shard=SOURCE/'transformer'/index[name]
                with safe_open(str(shard),framework='pt',device='cpu') as handle:
                    tensor=handle.get_tensor(name).pin_memory()
                assert tensor.dtype==torch.bfloat16, 'Original modulation precision changed'
                values[kind]=tensor
            self._host_weight=values['weight']
            self._host_bias=values['bias']
            event('i2v_adaln_original_projection_loaded',block=block,seconds=time.monotonic()-started,
                  bytes=sum(t.numel()*t.element_size() for t in values.values()))
    Modulation._study_load_original=load_original
    def build_missing(self,temb):
        cache=getattr(self,'_modulation_cache',None)
        if cache is not None and self._cache_key not in cache and getattr(self,'_host_weight',None) is None:
            load_original(self)
        return forward(self,temb)
    Modulation.forward=build_missing

def prefetch(generator):
    from concurrent.futures import ThreadPoolExecutor
    transformer=generator.executor.driver_worker.worker.pipeline.get_module('transformer')
    modules=[block.adaln_proj for block in transformer.transformer_blocks]
    if all(sum(len(key[0])==3 for key in module._modulation_cache)==8 for module in modules): return
    if all(getattr(module,'_host_weight',None) is not None for module in modules): return
    started=time.monotonic()
    event('i2v_adaln_projection_prefetch_started',threads=4,blocks=len(modules))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda module:module._study_load_original(),modules))
    event('i2v_adaln_projection_prefetch_completed',seconds=time.monotonic()-started)

def publish_tables(generator):
    import torch
    transformer=generator.executor.driver_worker.worker.pipeline.get_module('transformer')
    result={}
    complete=True
    entries=0
    for index,block in enumerate(transformer.transformer_blocks):
        table=block.adaln_proj._modulation_cache
        # Include only the exact FL2VA clock sets. Existing T2VA data stays unchanged.
        selected={repr(key):value.detach().cpu() for key,value in table.items()
                  if len(key[0])==3 and abs(key[0][-1]-.999)<1e-6}
        entries+=len(selected)
        if len(selected)!=8: complete=False
        result[index]=selected
    if not entries: return
    DESTINATION.mkdir(parents=True,exist_ok=True)
    local=ROOT/'adaln_tables_i2v.pt'
    torch.save(result,local)
    pending=DESTINATION/'adaln_tables_i2v.pending.pt'
    import shutil
    shutil.copyfile(local,pending)
    digest=sha(local)
    assert sha(pending)==digest
    receipt={'source_revision':REVISION,'entries':entries,'blocks':len(result),'complete':complete,
        'method':'Native MiniMaxH3AdaLayerNormModulation.forward with original checkpoint BF16 host weights; no approximation',
        'sha256':digest,'bytes':local.stat().st_size,'clock_sets':[ast.literal_eval(key) for key in result[0]],
        'created_at':time.time()}
    if complete:
        pending.replace(DESTINATION/'adaln_tables_i2v.pt')
        write_json(DESTINATION/'ready.json',receipt)
        event('i2v_adaln_table_ready',**receipt)
    else:
        write_json(DESTINATION/'partial.json',receipt)
        event('i2v_adaln_table_partial',entries=entries)

