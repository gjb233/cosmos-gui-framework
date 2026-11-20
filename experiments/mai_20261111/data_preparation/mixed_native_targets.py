"""Lossless direct-JSON target tokenization for the native training interface."""
import json
from mixed_prompt import FIELDS,ACTION_HELP
def tokenize_targets(tokenizer,texts,return_masks=False,max_tokens=4096):
 ids_all=[];weights_all=[]
 for text in texts:
  def unique(pairs):
   out={}
   for k,v in pairs:
    if k in out:raise ValueError('duplicate action field')
    out[k]=v
   return out
  obj=json.loads(text,object_pairs_hook=unique)
  if set(obj)!=set(FIELDS) or obj['action_type'] not in ACTION_HELP:raise ValueError('invalid action schema')
  ids=list(tokenizer(text,add_special_tokens=False)['input_ids'])
  eos=tokenizer.eos_token_id
  if eos is None:raise ValueError('missing native EOS')
  if tokenizer.decode(ids,skip_special_tokens=False)!=text:raise ValueError('non-lossless target tokens')
  ids.append(eos)
  if len(ids)>max_tokens:raise ValueError('target over budget; never truncate action')
  # Standard assistant-only token CE. Each sample has total target weight 1.
  weights=[1.0/len(ids)]*len(ids)
  ids_all.append(ids);weights_all.append(weights)
 return (ids_all,weights_all) if return_masks else ids_all
