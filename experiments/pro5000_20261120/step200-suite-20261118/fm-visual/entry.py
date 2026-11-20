import sys,os,json
from pathlib import Path
V=Path(__file__).parent;W=Path('/data/mai/training/gui-libra-awhigh-aclow-fm2-20261117')
sys.path[:0]=[str(V),str(W),'/data/mai/code/f2f8996','/data/mai/code/f2f8996/experiments/mai_20261111/native_template']
import vis_backend as b
original=b.install
samples=json.loads(Path('/data/mai/evaluation/step200-suite-20261118/fm-visual/samples.json').read_text())
def install(records,draws):
 chosen=[dict(draw=i,id=r['id'],source=r['source'],level='high' if 'high' in r['views'] else 'low',variant='direct') for i,r in enumerate(samples)]
 original(records,chosen*16)
 from cosmos_framework.gui_mot.native_model import GuiJointModel
 from cosmos_framework.gui_mot import native_model as nm
 import ast,textwrap
 source=Path(nm.__file__).read_text()
 cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='GuiJointModel')
 method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_prepare_gui_prefix')
 ns=dict(vars(nm));exec(textwrap.dedent(ast.get_source_segment(source,method)),ns)
 base_prepare=ns['_prepare_gui_prefix']
 def prepare(self,batch):
  result=base_prepare(self,batch)
  captions=batch['ai_caption']
  while isinstance(captions,list) and len(captions)==1 and isinstance(captions[0],list):captions=captions[0]
  if isinstance(captions,str):captions=[captions]
  self._history_fm_contexts=[json.loads(x) for x in captions]
  self._history_fm_context=self._history_fm_contexts[0]
  return result
 GuiJointModel._prepare_gui_prefix=prepare
 def inference_tokens(self,batch,has_negative_prompt,caption_groups=None):
  assert not has_negative_prompt
  tokens=self._prepare_gui_prefix(batch);assert len(tokens)==1
  return tokens,tokens
 GuiJointModel._get_inference_text_tokens=inference_tokens
b.install=install
import zero2_hook
import zero2_micro2_adapter as zadapter
from cosmos_framework.gui_mot import multimodal as mm
mm.hybrid_ar_forward=zadapter.single
from fm_vis_hook import train
from cosmos_framework.trainer import ImaginaireTrainer
ImaginaireTrainer.train=train
b.main()
