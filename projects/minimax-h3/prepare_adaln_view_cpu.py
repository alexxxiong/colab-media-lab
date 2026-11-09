"""Keep original Drive weights; save bit-identical active shards for fixed 8-step tables."""
import hashlib
import json
from pathlib import Path
import re
import struct
import time

MODEL=Path('/content/drive/MyDrive/Colab-Lab/models/base/MiniMax-H3/fasth3-v2-nvfp4/9d0d0d3ae69365aded83fab5562f00f9f06c12aa')
WORK=Path('/content/colab-lab/h3-preparation')
STATUS=WORK/'adaln-view-status.json'

def header(path):
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>64*1024**2:
            raise ValueError('Invalid safetensors header')
        return 8+length,json.loads(stream.read(length))

def repack(source,destination):
    base,original=header(source)
    keep={}
    removed=[]
    for key,value in original.items():
        if key=='__metadata__':
            continue
        if re.fullmatch(r'transformer_blocks\.\d+\.adaln_proj\.linear\.(weight|bias)',key):
            removed.append(key)
        else:
            keep[key]=dict(value)
    offset=0
    for value in keep.values():
        begin,end=value['data_offsets']
        value['data_offsets']=[offset,offset+end-begin]
        offset+=end-begin
    packed={'__metadata__':original.get('__metadata__',{}),**keep}
    encoded=json.dumps(packed,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    temporary=destination.with_suffix('.uploading')
    destination.parent.mkdir(parents=True,exist_ok=True)
    hashes={}
    with source.open('rb') as reader,temporary.open('wb') as writer:
        writer.write(struct.pack('<Q',len(encoded)))
        writer.write(encoded)
        for key in keep:
            begin,end=original[key]['data_offsets']
            reader.seek(base+begin)
            digest=hashlib.sha256()
            remaining=end-begin
            while remaining:
                data=reader.read(min(remaining,8*1024**2))
                if not data:
                    raise IOError('Truncated source tensor')
                writer.write(data)
                digest.update(data)
                remaining-=len(data)
            hashes[key]=digest.hexdigest()
    # Verify tensor bytes after the write. No quantization or numeric changes.
    new_base,new_header=header(temporary)
    with temporary.open('rb') as reader:
        for key,expected in hashes.items():
            begin,end=new_header[key]['data_offsets']
            reader.seek(new_base+begin)
            digest=hashlib.sha256()
            remaining=end-begin
            while remaining:
                data=reader.read(min(remaining,8*1024**2))
                if not data:
                    raise IOError('Truncated derived tensor')
                digest.update(data)
                remaining-=len(data)
            if digest.hexdigest()!=expected:
                raise IOError('Active tensor bytes changed')
    temporary.replace(destination)
    with destination.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'source_bytes':source.stat().st_size,'bytes':destination.stat().st_size,
            'retained_tensors':len(keep),'removed_tensors':removed,
            'sha256':digest}

def main():
    started=time.time()
    STATUS.write_text(json.dumps({'state':'waiting','started':started}))
    while not (MODEL/'_COLAB_LAB_READY.json').is_file():
        if time.time()-started>3600:
            raise TimeoutError('Model preparation did not finish within the CPU window')
        time.sleep(10)
    STATUS.write_text(json.dumps({'state':'running','started':started}))
    assert (MODEL/'transformer/adaln_tables.pt').is_file()
    folder=MODEL/'_COLAB_LAB_ADALN_VIEW'
    folder.mkdir(exist_ok=True)
    files={}
    for source in sorted((MODEL/'transformer').glob('diffusion_pytorch_model-*-of-*.safetensors')):
        relative=source.relative_to(MODEL).as_posix()
        files[relative]=repack(source,folder/relative)
    if sum(len(v['removed_tensors']) for v in files.values())!=100:
        raise RuntimeError('Expected exactly 50 AdaLN projection weights and biases')
    receipt={'source_revision':MODEL.name,'required_table':'transformer/adaln_tables.pt',
             'schedule':'8-forward V2 only','active_tensors_bit_identical':True,'files':files,
             'bytes_avoided':sum(v['source_bytes']-v['bytes'] for v in files.values())}
    (folder/'ready.json').write_text(json.dumps(receipt,indent=2))
    STATUS.write_text(json.dumps({'state':'passed','started':started,'finished':time.time(),
                                 'bytes_avoided':receipt['bytes_avoided'],'shards':len(files)}))

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        STATUS.write_text(json.dumps({'state':'failed','error':type(exc).__name__,'message':str(exc)[:300]}))
        raise
