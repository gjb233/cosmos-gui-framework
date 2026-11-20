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
 def inference_tokens(self,batch,has_negative_prompt,caption_groups=None):
  assert not has_negative_prompt
  tokens=self._prepare_gui_prefix(batch);assert len(tokens)==1
  return tokens,tokens
 GuiJointModel._get_inference_text_tokens=inference_tokens
b.install=install
import zero2_hook
from fm_vis_hook import train
from cosmos_framework.trainer import ImaginaireTrainer
ImaginaireTrainer.train=train
b.main()
