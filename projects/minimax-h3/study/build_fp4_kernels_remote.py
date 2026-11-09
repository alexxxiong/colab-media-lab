"""Build only the two missing pinned SM120 FP4 extensions, not all kernels."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
sys.path.insert(0,'/content/colab-lab/whale-study')
from study_runtime import ROOT,PROJECT,event,sha,write_json

source=Path('/content/colab-lab/h3-environment/FastVideo')
kernel=source/'fastvideo-kernel'
build=Path('/content/colab-lab/fp4-only-build-ce970eb71521')
overlay=Path('/content/colab-lab/h3-kernel-overlay')
cache=PROJECT/'cache'
log_path=ROOT/'fp4-kernel-build.log'
env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
         TORCH_CUDA_ARCH_LIST='12.0a',CUDA_HOME='/usr/local/cuda',CUDACXX='/usr/local/cuda/bin/nvcc',
         MAX_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2')
stopped=threading.Event()

def publish_log():
    while not stopped.wait(15):
        if log_path.exists(): shutil.copyfile(log_path,cache/'fp4-kernel-build-ce970eb71521.log')
threading.Thread(target=publish_log,daemon=True).start()

def run(command):
    event('fp4_kernel_build_command',command=command)
    with log_path.open('a',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:
            log.write(line)
            print(line,end='',flush=True)
            if line.strip(): event('fp4_kernel_build_output',line=line.strip()[:500])
        code=process.wait()
    if code: raise RuntimeError('Build subprocess exited '+str(code))

try:
    import torch
    assert torch.__version__.split('+')[0]=='2.12.0'
    revision='ce970eb715215c79a169658d7aa6c8a96e64891e'
    manifest=json.loads((ROOT/'kernel-source-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha(kernel/name)==digest, 'Kernel source checksum mismatch: '+name
    event('fp4_kernel_build_started',source_revision=revision)
    cutlass=kernel/'include/cutlass'
    cutlass_revision=manifest['cutlass_revision']
    if not (cutlass/'include/cutlass/cutlass.h').is_file():
        event('pinned_cutlass_download_started',revision=cutlass_revision)
        archive=ROOT/'pinned-cutlass-source.tar.gz'
        with urllib.request.urlopen('https://codeload.github.com/NVIDIA/cutlass/tar.gz/'+cutlass_revision,timeout=60) as response,archive.open('wb') as output:
            shutil.copyfileobj(response,output,1024**2)
        event('pinned_cutlass_download_completed',bytes=archive.stat().st_size,sha256=sha(archive))
        cutlass.mkdir(parents=True,exist_ok=True)
        prefix='cutlass-'+cutlass_revision
        run(['tar','-xzf',str(archive),'--strip-components=1','-C',str(cutlass),
             prefix+'/include',prefix+'/tools/util/include'])
    from torch.utils import cmake_prefix_path
    # The root packaging target requests a Python stable-ABI target that a
    # direct CMake configure does not create. Build only the unmodified FP4
    # target definitions; neither needs that unrelated packaging target.
    native_cmake=(kernel/'CMakeLists.txt').read_text()
    begin=native_cmake.index('if(ENABLE_ATTN_QAT_INFER)\n    set(ATTN_QAT_INFER_DIR')
    end=native_cmake.index('    install(TARGETS fp4attn_cuda',begin)
    definitions=native_cmake[begin:end].replace('${CMAKE_SOURCE_DIR}','${H3_KERNEL_SOURCE_DIR}')+'\nendif()\n'
    wrapper=ROOT/'fp4-cmake-only'
    wrapper.mkdir(exist_ok=True)
    link=wrapper/'attn_qat_infer'
    if not link.exists(): link.symlink_to(kernel/'attn_qat_infer',target_is_directory=True)
    assert link.resolve()==(kernel/'attn_qat_infer').resolve()
    header='''cmake_minimum_required(VERSION 3.26)
project(h3_fp4_only LANGUAGES CXX CUDA)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
set(ENABLE_ATTN_QAT_INFER ON)
set(FASTVIDEO_FP4_CUDA_ARCHS 120a)
'''
    header+='set(H3_KERNEL_SOURCE_DIR "'+str(kernel)+'")\n'
    header+='set(TORCH_PYTHON_LIBRARY_PATH "'+str(Path(torch.__file__).parent/'lib/libtorch_python.so')+'")\n'
    (wrapper/'CMakeLists.txt').write_text(header+definitions)
    run(['cmake','-S',str(wrapper),'-B',str(build),'-G','Ninja',
         '-DCMAKE_BUILD_TYPE=Release','-DPython_EXECUTABLE='+sys.executable,
         '-DCMAKE_PREFIX_PATH='+cmake_prefix_path,'-DSKBUILD_SABI_VERSION=3.10',
         '-DFASTVIDEO_KERNEL_BUILD_TK=OFF','-DFASTVIDEO_KERNEL_BUILD_ULYSSES_A2A=OFF',
         '-DFASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER=ON','-DTORCH_CUDA_ARCH_LIST=12.0a'])
    run(['cmake','--build',str(build),'--parallel','2','--target','fp4attn_cuda','fp4quant_cuda'])
    overlay.mkdir(exist_ok=True)
    for prefix in ('fp4attn_cuda','fp4quant_cuda'):
        paths=list(build.rglob(prefix+'*.so'))
        assert len(paths)==1,(prefix,paths)
        shutil.copyfile(paths[0],overlay/paths[0].name)
    package=overlay/'attn_qat_infer'
    package.mkdir(exist_ok=True)
    for name in ('__init__.py','api.py'): shutil.copyfile(kernel/'attn_qat_infer'/name,package/name)
    # Validate the actual compiled ABI and matching Python API in a fresh process.
    check='import sys,torch;sys.path.insert(0,'+repr(str(overlay))+');import fp4attn_cuda,fp4quant_cuda;import attn_qat_infer.api as a;assert hasattr(fp4attn_cuda,"fwd_sparse");assert hasattr(a,"vsa_tile_mask_to_fp4_blocks");print("PINNED_FP4_API_ABI_VERIFIED")'
    run([sys.executable,'-c',check])
    receipt={'source_revision':revision,'torch':torch.__version__,'architecture':'sm_120a',
             'cutlass_revision':cutlass_revision,'build_wrapper_sha256':sha(wrapper/'CMakeLists.txt'),
             'files':{str(p.relative_to(overlay)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in overlay.rglob('*') if p.is_file() and '__pycache__' not in str(p)}}
    write_json(overlay/'ready.json',receipt)
    archive=ROOT/'fp4-kernel-overlay-ce970eb71521-sm120-py312.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle: bundle.add(overlay,arcname='h3-kernel-overlay',filter=lambda x:None if '__pycache__' in x.name else x)
    target=cache/archive.name
    shutil.copyfile(archive,target)
    assert sha(target)==sha(archive)
    receipt.update(archive=str(target),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    write_json(cache/'fp4-kernel-overlay-ready.json',receipt)
    event('fp4_kernel_build_completed',**receipt)
except BaseException as exc:
    write_json(ROOT/'fp4-kernel-build-error.json',{'error':str(exc),'at':time.time()})
    event('fp4_kernel_build_failed',error=str(exc))
    raise
finally:
    stopped.set()
    if log_path.exists(): shutil.copyfile(log_path,cache/'fp4-kernel-build-ce970eb71521.log')
