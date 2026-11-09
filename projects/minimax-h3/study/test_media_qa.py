"""Verify acceptance rejects broken geometry/audio using real CPU-encoded media."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
import av
import numpy as np

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'worker.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='media_qa')
namespace={}
exec(compile(ast.Module(body=[function],type_ignores=[]),'worker.py:media_qa','exec'),namespace)
media_qa=namespace['media_qa']

def make_media(path,audio=True):
    with av.open(str(path),'w') as output:
        video=output.add_stream('libx264',rate=24)
        video.width=480
        video.height=832
        video.pix_fmt='yuv420p'
        video.options={'preset':'ultrafast','crf':'28'}
        if audio:
            stream=output.add_stream('aac',rate=32000)
            stream.layout='stereo'
            stream.time_base=Fraction(1,32000)
        for index in range(124):
            frame=av.VideoFrame.from_ndarray(np.full((832,480,3),index%200,dtype=np.uint8),format='rgb24')
            for packet in video.encode(frame): output.mux(packet)
        for packet in video.encode(): output.mux(packet)
        if audio:
            count=0
            for index in range(162):
                signal=(.1*np.sin(2*np.pi*440*(np.arange(1024)+count)/32000)).astype('float32')
                frame=av.AudioFrame.from_ndarray(np.stack((signal,signal)),format='fltp',layout='stereo')
                frame.sample_rate=32000
                frame.time_base=Fraction(1,32000)
                frame.pts=count
                count+=1024
                for packet in stream.encode(frame): output.mux(packet)
            for packet in stream.encode(): output.mux(packet)

class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.good=Path(cls.temporary.name)/'good.mp4'
        make_media(cls.good)
        cls.case={'width':480,'height':832,'frames':124,'actual_seconds':124/24}

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_actual_video_and_audio_decode(self):
        result=media_qa(self.good,self.case)
        self.assertEqual(result['frames'],124)
        self.assertEqual(result['audio_channels'],2)
        self.assertGreater(result['audio_rms'],0)

    def test_wrong_size_is_rejected(self):
        with self.assertRaises(AssertionError): media_qa(self.good,dict(self.case,width=768))

    def test_wrong_frame_count_is_rejected(self):
        with self.assertRaises(AssertionError): media_qa(self.good,dict(self.case,frames=243))

    def test_silent_missing_track_is_rejected(self):
        path=Path(self.temporary.name)/'no-audio.mp4'
        make_media(path,audio=False)
        with self.assertRaises(AssertionError): media_qa(path,self.case)

if __name__=='__main__': unittest.main()
