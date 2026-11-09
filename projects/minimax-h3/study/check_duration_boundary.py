"""Execute only the pinned, pure frame validators on the host CPU."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
source=root/'fastvideo-source/fastvideo/pipelines/basic/minimax_h3'
names={'MINIMAX_H3_FPS','MINIMAX_H3_MIN_DURATION','MINIMAX_H3_MAX_DURATION',
       'MINIMAX_H3_FRAMES_PER_CHUNK','MINIMAX_H3_LATENTS_PER_CHUNK',
       'MINIMAX_H3_MIN_ALIGNED_FRAMES','MINIMAX_H3_MAX_ALIGNED_FRAMES'}
nodes=[]
for node in ast.parse((source/'packing.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets): nodes.append(node)
    if isinstance(node,ast.FunctionDef) and node.name=='align_num_frames': nodes.append(node)
function=next(n for n in ast.parse((source/'stages/minimax_h3_input_preparation.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='resolve_target_num_frames')
nodes.append(function)
namespace={}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-h3-duration-validator','exec'),namespace)
results=[]
for seconds,frames in ((5,124),(10,243),(15,362),(20,480),(20,481)):
    row={'seconds_bucket':seconds,'requested_frames':frames}
    try: row.update(accepted=True,aligned_frames=namespace['resolve_target_num_frames'](frames))
    except ValueError as exc: row.update(accepted=False,error=str(exc))
    results.append(row)
assert [r['accepted'] for r in results]==[True,True,True,False,False]
report={'method':'Host CPU execution of extracted pure validation functions; no model/GPU inference',
        'source_revision':'ce970eb715215c79a169658d7aa6c8a96e64891e','results':results,
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (source/'packing.py',source/'stages/minimax_h3_input_preparation.py')}}
(root/'duration-boundary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,ensure_ascii=False))
