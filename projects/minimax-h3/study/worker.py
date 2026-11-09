"""Persistent one-GPU worker. Every request encodes its first frame and prompt again."""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
import zipfile

from study_runtime import ROOT,ASSETS,event,freeze_assets,publish,sha,stage_model,write_json
from resource_sampler import Sampler

OPTIONS=json.loads((ROOT/'session-options.json').read_text())
SESSION=OPTIONS['session']
for folder in ('commands','results'): (ROOT/folder).mkdir(exist_ok=True)
STAGES=[]
WORKER_METRICS={}
MEDIA_SAVE=[]

def state(status,**data):
    write_json(ROOT/'state.json',{'at':time.time(),'status':status,'pid':os.getpid(),**data})

def media_qa(path,case,create_contact=True):
    import av
    import numpy as np
    with av.open(str(path)) as container:
        assert len(container.streams.video)==1 and len(container.streams.audio)==1,'Missing video/audio stream'
        stream=container.streams.video[0]
        assert (stream.width,stream.height)==(case['width'],case['height']), 'Video dimensions changed'
        assert abs(float(stream.average_rate)-24)<.01,'Wrong frame rate'
        frames=0
        thumbnails=[]
        wanted={round((case['frames']-1)*x/8) for x in range(9)}
        for frame in container.decode(video=0):
            if create_contact and frames in wanted:
                thumb=frame.to_image()
                thumb.thumbnail((320,560))
                thumbnails.append(thumb)
            frames+=1
        assert frames==case['frames'],f'Expected {case["frames"]} frames, decoded {frames}'
        duration=float(container.duration/av.time_base)
        assert abs(duration-case['actual_seconds'])<.2,'Wrong video duration'
    if create_contact:
        from PIL import Image,ImageDraw
        sheet=Image.new('RGB',(320*3,600*3),'white')
        for index,thumb in enumerate(thumbnails):
            x,y=(index%3)*320,(index//3)*600
            sheet.paste(thumb,(x+(320-thumb.width)//2,y+25))
            ImageDraw.Draw(sheet).text((x+8,y+5),f'{index/8*case["actual_seconds"]:.2f}s',fill='black')
        sheet.save(path.with_suffix('.contact.png'))
    audio_samples=0
    sumsq=0.0
    peak=0.0
    with av.open(str(path)) as container:
        audio=container.streams.audio[0]
        rate=audio.codec_context.sample_rate
        channels=audio.codec_context.channels
        for frame in container.decode(audio=0):
            raw=frame.to_ndarray()
            data=raw.astype('float64')
            if raw.dtype.kind in 'iu': data/=max(abs(__import__('numpy').iinfo(raw.dtype).min),__import__('numpy').iinfo(raw.dtype).max)
            audio_samples+=frame.samples
            sumsq+=float((data*data).sum())
            peak=max(peak,float(np.max(np.abs(data))))
    assert audio_samples>0 and sumsq>0,'Generated audio is missing or silent'
    return dict(width=case['width'],height=case['height'],fps=24,frames=frames,
        duration_seconds=duration,video_duration_seconds=frames/24,container_duration_seconds=duration,
        audio_sample_rate=rate,audio_channels=channels,
        audio_samples=audio_samples,audio_peak=peak,audio_rms=(sumsq/max(1,audio_samples*channels))**.5,
        decoded=True,content_review='pending')

def instrument():
    import torch
    from fastvideo.pipelines.stages.base import PipelineStage
    from fastvideo.worker.gpu_worker import Worker
    from fastvideo import VideoGenerator
    def save_wrapper(original,name):
        def measured(cls,*args,**kwargs):
            start=time.monotonic()
            try: return original(*args,**kwargs)
            finally: MEDIA_SAVE.append({'method':name,'seconds':time.monotonic()-start,'pid':os.getpid()})
        return classmethod(measured)
    for name in ('_save_video_with_audio_ffmpeg_pipe','_save_video_with_audio_single_pass','_mux_audio'):
        setattr(VideoGenerator,name,save_wrapper(getattr(VideoGenerator,name),name))
    original=PipelineStage.__call__
    def timed_stage(self,*args,**kwargs):
        torch.cuda.synchronize()
        start=time.monotonic()
        name=self.__class__.__name__
        event('stage_started',name=name)
        status='passed'
        try: return original(self,*args,**kwargs)
        except BaseException:
            status='failed'
            raise
        finally:
            torch.cuda.synchronize()
            record={'name':name,'seconds':time.monotonic()-start,'status':status,'pid':os.getpid(),
                    'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            STAGES.append(record)
            event('stage_completed',**record)
    PipelineStage.__call__=timed_stage
    execute=Worker.execute_forward
    def timed_forward(self,*args,**kwargs):
        transformer=self.pipeline.get_module('transformer')
        if transformer is not None and hasattr(transformer,'_text_cache'):
            transformer._text_cache=None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        WORKER_METRICS.clear()
        try: return execute(self,*args,**kwargs)
        finally:
            torch.cuda.synchronize()
            WORKER_METRICS.update(pid=os.getpid(),seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Worker.execute_forward=timed_forward

def make_generator(model,profile):
    from fastvideo import VideoGenerator
    compiled=profile in ('vae-compiled','vae-compiled-cpu-offload')
    offload=profile in ('cpu-offload','vae-compiled-cpu-offload')
    settings={
        'model_path':str(model),
        'engine':{'num_gpus':1,'execution_backend':'uni',
            'quantization':{'transformer_quant':'NVFP4','layer_profile':'h3_dit_vsa'},
            'offload':{'dit':False,'dit_layerwise':False,'text_encoder':offload,'vae':offload,
                       'pin_cpu_memory':False,'lazy_module_load':False},
            'compile':{'enabled':False,'vae_enabled':compiled,'audio_vae_enabled':False,'text_encoder_enabled':False}},
        'pipeline':{'experimental':{'attention_backend':'VIDEO_SPARSE_ATTN_H3','VSA_sparsity':.8,'VSA_tile_size':64}}}
    event('framework_initialization_started',profile=profile,settings=settings)
    state('loading',profile=profile)
    start=time.monotonic()
    gen=VideoGenerator.from_config(settings)
    event('framework_initialization_completed',profile=profile,seconds=time.monotonic()-start,
          executor=type(gen.executor).__name__,vae_compile_requested=compiled,
          vae_compile_flag=gen.fastvideo_args.enable_torch_compile_vae)
    assert gen.fastvideo_args.enable_torch_compile_vae==compiled
    assert type(gen.executor).__name__=='UniprocExecutor','Instrumentation requires the actual in-process GPU worker'
    return gen

def generate(gen,job,case,iteration,assets):
    import torch
    from PIL import Image
    from fastvideo.api import GenerationRequest,InputConfig,OutputConfig,SamplingConfig
    seed=job.get('seed',42)
    label=f'{job["id"]}-{case["id"]}-s{seed}-r{iteration}'
    output=ROOT/'results'/f'{label}.mp4'
    if output.exists(): raise RuntimeError('Refusing to reuse an existing output as a new sample')
    metric={'id':label,'command':job['id'],'category':job.get('category','research'),
            'case':case,'seed':seed,'iteration':iteration,'profile':job.get('profile','baseline'),
            'started_at':time.time(),'passed':False,'mode':'i2v','content_usable':None,
            'model_revision':'9d0d0d3ae69365aded83fab5562f00f9f06c12aa',
            'runtime_git_revision':'ce970eb715215c79a169658d7aa6c8a96e64891e',
            'assets_version':OPTIONS.get('assets_version','whale-v1'),
            'cache_policy':{'output_cache':False,'reference_memo_entries':0,'cross_request_text_features':False,
                            'model_and_jit_caches':True},
            'input_sha256':sha(assets/case['image'])}
    state('generating',request=label)
    event('request_started',request=label)
    STAGES.clear()
    MEDIA_SAVE.clear()
    WORKER_METRICS.clear()
    start=time.monotonic()
    try:
        if OPTIONS.get('build_i2v_adaln'):
            from adaln_i2v import prefetch
            preparation_start=time.monotonic()
            prefetch(gen)
            metric['i2v_adaln_preparation_seconds']=time.monotonic()-preparation_start
        image_start=time.monotonic()
        image=Image.open(assets/case['image']).convert('RGB')
        metric['input_image_open_seconds']=time.monotonic()-image_start
        generation_start=time.monotonic()
        result=gen.generate(GenerationRequest(prompt=case['prompt'],negative_prompt='',
            inputs=InputConfig(pil_image=image,last_image=None),
            sampling=SamplingConfig(height=case['height'],width=case['width'],num_frames=case['frames'],
                fps=24,num_inference_steps=9,guidance_scale=1.,batch_cfg=False,seed=seed),
            output=OutputConfig(output_path=str(output),save_video=True,return_frames=False)))
        torch.cuda.synchronize()
        metric['generation_seconds']=time.monotonic()-generation_start
        actual=Path(result.video_path or output)
        assert actual.is_file() and actual.stat().st_size>0,'No nonempty media artifact'
        if actual!=output: shutil.copyfile(actual,output)
        qa_start=time.monotonic()
        metric['media_qa']=media_qa(output,case,create_contact=job.get('category','research')=='research')
        metric['media_qa_seconds']=time.monotonic()-qa_start
        metric['video_sha256']=sha(output)
        metric['video_bytes']=output.stat().st_size
        info=getattr(result,'logging_info',None)
        metric['fastvideo_logging']=getattr(info,'__dict__',str(info))
        metric['passed']=True
    except BaseException as exc:
        metric.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                      oom=isinstance(exc,torch.OutOfMemoryError) or 'out of memory' in str(exc).lower())
        traceback.print_exc()
    finally:
        if OPTIONS.get('build_i2v_adaln'):
            try:
                from adaln_i2v import publish_tables
                publish_tables(gen)
            except Exception:
                traceback.print_exc()
        metric['stages']=list(STAGES)
        metric['media_encode_mux']=list(MEDIA_SAVE)
        metric['actual_worker']=dict(WORKER_METRICS)
        metric['finished_at']=time.time()
        metric['request_seconds']=time.monotonic()-start
        write_json(ROOT/'results'/f'{label}.json',metric)
        publish_start=time.monotonic()
        publish(SESSION)
        metric['publish_seconds']=time.monotonic()-publish_start
        write_json(ROOT/'results'/f'{label}.json',metric)
        publish(SESSION)
        event('request_completed',request=label,passed=metric['passed'],seconds=metric['request_seconds'],oom=metric.get('oom',False))
    return metric

def package_results(job):
    """Snapshot evidence into bounded transfer chunks without re-encoding media."""
    event('artifact_packaging_started',command=job['id'])
    state('packaging')
    snapshot=ROOT/('snapshot-'+job['id'])
    snapshot.mkdir()
    relative=[]
    for name in ('events.jsonl','resources.jsonl','startup-resources.jsonl','worker.log','stdout.log','stderr.log'):
        source=ROOT/name
        if source.is_file():
            shutil.copyfile(source,snapshot/name)
            relative.append(Path(name))
    for path in sorted((ROOT/'results').iterdir()):
        if path.is_file():
            destination=snapshot/'results'/path.name
            destination.parent.mkdir(exist_ok=True)
            # Result files are immutable after their command completes.
            destination.hardlink_to(path)
            relative.append(Path('results')/path.name)
    folder=ROOT/'bundles'
    folder.mkdir(exist_ok=True)
    chunks=[]
    part=[]
    total=0
    manifest={'command':job['id'],'files':{},'chunks':[]}
    def finish(items):
        if not items: return
        name=f'{job["id"]}-part{len(chunks):02d}.zip'
        pending=ROOT/(name+'.building')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_STORED) as archive:
            for rel in items: archive.write(snapshot/rel,str(rel))
        pending.rename(folder/name)
        record={'name':name,'bytes':(folder/name).stat().st_size,'sha256':sha(folder/name)}
        chunks.append(record)
        event('artifact_chunk_completed',**record)
    for rel in relative:
        size=(snapshot/rel).stat().st_size
        if part and total+size>64*1024**2:
            finish(part)
            part=[]
            total=0
        part.append(rel)
        total+=size
        manifest['files'][str(rel)]={'bytes':size,'sha256':sha(snapshot/rel)}
    finish(part)
    manifest['chunks']=chunks
    write_json(folder/(job['id']+'-manifest.json'),manifest)
    publish(SESSION)
    write_json(ROOT/'results'/(job['id']+'-command.json'),{'completed':True,'packaged':True,'manifest':str(folder/(job['id']+'-manifest.json'))})
    state('idle',last_command=job['id'],packaged=True)
    event('artifact_packaging_completed',command=job['id'],chunks=len(chunks))

def main():
    sampler=Sampler(ROOT/'resources.jsonl',os.getpid()).start()
    publisher_stop=threading.Event()
    def publisher():
        while not publisher_stop.wait(15):
            try: publish(SESSION)
            except Exception: traceback.print_exc()
    threading.Thread(target=publisher,daemon=True).start()
    generator=None
    current_profile=None
    exit_code=0
    try:
        event('worker_started',options=OPTIONS)
        state('staging')
        model=stage_model()
        assets=freeze_assets()
        cases={c['id']:c for c in json.loads((assets/'cases.json').read_text())}
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'hf-cache'),
            FASTVIDEO_H3_VSA_FP4='1',FASTVIDEO_MINIMAX_H3_FUSIONS='all',FASTVIDEO_NVFP4_MM_BACKEND='cutlass',
            FASTVIDEO_INFERENCE_TORCH_COMPILE='0',FASTVIDEO_STAGE_LOGGING='0',FASTVIDEO_H3_REF2VA_MEMO_ENTRIES='0',
            FASTVIDEO_H3_ADALN_TABLE=str(model/'transformer/adaln_tables.pt'))
        import torch
        assert torch.__version__.split('+')[0]=='2.12.0'
        import importlib.metadata
        versions={}
        for package in ('torch','flashinfer-python','nvidia-cudnn-frontend','cutlass-dsl',
                        'quack-kernels','transformers','diffusers','av','fastvideo-kernel'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package]='distribution metadata absent'
        write_json(ROOT/'runtime-versions.json',{'python':__import__('sys').version,'cuda':torch.version.cuda,
                   'packages':versions})
        from kernel_overlay import apply
        apply()
        from adaln_i2v import DESTINATION
        receipt_path=DESTINATION/'ready.json'
        if receipt_path.is_file():
            receipt=json.loads(receipt_path.read_text())
            assert receipt['complete'] and receipt['source_revision']==model.name
            destination=model/'transformer/adaln_tables_i2v.pt'
            from study_runtime import copy_verified
            copy_verified(DESTINATION/'adaln_tables_i2v.pt',destination,receipt['sha256'])
            os.environ['FASTVIDEO_H3_ADALN_TABLE']=str(destination)
            event('i2v_adaln_table_verified',sha256=receipt['sha256'])
        elif not OPTIONS.get('build_i2v_adaln'):
            raise RuntimeError('Complete verified I2V modulation table required before formal sampling')
        elif (DESTINATION/'partial.json').is_file():
            partial_record=json.loads((DESTINATION/'partial.json').read_text())
            partial_path=DESTINATION/'adaln_tables_i2v.pending.pt'
            assert sha(partial_path)==partial_record['sha256']
            base=torch.load(model/'transformer/adaln_tables.pt',map_location='cpu',weights_only=True)
            partial=torch.load(partial_path,map_location='cpu',weights_only=True)
            for index,items in partial.items(): base[index].update(items)
            merged=ROOT/'adaln_tables_research_resume.pt'
            torch.save(base,merged)
            os.environ['FASTVIDEO_H3_ADALN_TABLE']=str(merged)
            event('partial_i2v_adaln_table_reused',entries=partial_record['entries'])
        if OPTIONS.get('build_i2v_adaln'):
            from adaln_i2v import install
            install()
        instrument()
        event('worker_ready',torch=torch.__version__)
        state('idle')
        processed=set()
        while True:
            jobs=sorted((ROOT/'commands').glob('*.json'))
            for path in jobs:
                if path.name in processed: continue
                job=json.loads(path.read_text())
                processed.add(path.name)
                if job['action']=='stop':
                    event('worker_stop_requested')
                    return 0
                if job['action']=='package':
                    package_results(job)
                    continue
                profile=job.get('profile','baseline')
                try:
                    if current_profile!=profile:
                        if generator:
                            generator.shutdown()
                            del generator
                            generator=None
                            gc.collect()
                            if current_profile in ('vae-compiled','vae-compiled-cpu-offload'):
                                torch._dynamo.reset()
                                gc.collect()
                            torch.cuda.empty_cache()
                            event('profile_memory_after_shutdown',previous_profile=current_profile,
                                  allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
                        generator=make_generator(model,profile)
                        current_profile=profile
                    results=[]
                    for case_id in job['cases']:
                        for iteration in range(job.get('repeats',1)):
                            item=generate(generator,job,cases[case_id],iteration,assets)
                            results.append({'id':item['id'],'passed':item['passed'],'oom':item.get('oom',False)})
                            # OOM is a capacity observation. Do not waste five unchanged retries.
                            if not item['passed']:
                                if item.get('oom'):
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                break
                        # Capacity failures may advance to another specification.
                        # An unexplained software/media error stops the batch;
                        # do not repeat the same broken path across all cases.
                        if not results[-1]['passed'] and (job.get('stop_on_failure',True) or not results[-1].get('oom')): break
                    write_json(ROOT/'results'/f'{job["id"]}-command.json',{'completed':True,'results':results})
                    state('idle',last_command=job['id'],results=results)
                    event('command_completed',command=job['id'],results=results)
                except BaseException as exc:
                    traceback.print_exc()
                    write_json(ROOT/'results'/f'{job["id"]}-command.json',
                        {'completed':True,'failed':True,'error':str(exc),'traceback':traceback.format_exc()})
                    state('command_failed',last_command=job['id'],error=str(exc))
                    event('command_failed',command=job['id'],error=str(exc))
                publish(SESSION)
            time.sleep(.5)
    except BaseException as exc:
        exit_code=1
        traceback.print_exc()
        state('fatal',error=str(exc),traceback=traceback.format_exc())
        event('worker_fatal',error=str(exc))
    finally:
        if generator:
            try: generator.shutdown()
            except Exception: traceback.print_exc()
        sampler.stop()
        publisher_stop.set()
        event('worker_exit')
        publish(SESSION)
    return exit_code

if __name__=='__main__': raise SystemExit(main())
