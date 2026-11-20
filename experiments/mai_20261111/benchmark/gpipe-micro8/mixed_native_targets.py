from native_protocol import parse
def tokenize_targets(tokenizer,texts,return_masks=False,max_tokens=4096):
 ids_all=[];weights_all=[]
 for text in texts:
  parse(text)
  ids=list(tokenizer(text,add_special_tokens=False)['input_ids'])
  assert tokenizer.decode(ids,skip_special_tokens=False)==text
  assert tokenizer.eos_token_id is not None
  ids.append(tokenizer.eos_token_id)
  assert len(ids)<=max_tokens,'never truncate action'
  ids_all.append(ids);weights_all.append([1.0/len(ids)]*len(ids))
 return (ids_all,weights_all) if return_masks else ids_all
