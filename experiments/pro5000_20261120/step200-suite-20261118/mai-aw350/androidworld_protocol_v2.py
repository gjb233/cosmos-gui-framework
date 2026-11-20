"""Corrected no-thinking prompt; parse one validated action, ignoring reasoning."""
import json,re,sys
sys.path.insert(0,'/data/mai/evaluation/assets-20261113/grounding')
from native_protocol import SYSTEM as ORIGINAL_SYSTEM,validate,ac_text
OLD='- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.'
assert ORIGINAL_SYSTEM.count(OLD)==1
SYSTEM=ORIGINAL_SYSTEM.replace(OLD,'- Follow the Action Space strictly. Return only one JSON object inside <tool_call></tool_call>; do not output a thinking process.')
def parse(text):
 text=text.strip()
 # Ignore a complete leading reasoning section only, not arbitrary extra text.
 text=re.sub(r'^<thinking>.*?</thinking>\s*','',text,count=1,flags=re.S)
 match=re.fullmatch(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
 if not match:raise ValueError('Expected exactly one complete tool_call')
 return validate(json.loads(match.group(1)))
def test():
 a={'name':'mobile_use','arguments':{'action':'click','coordinate':[499,898]}}
 block='<tool_call>\n'+json.dumps(a)+'\n</tool_call>'
 assert parse(block)==a
 assert parse('<thinking>reasoning</thinking>\n'+block)==a
 assert parse('  '+block+'  ')==a
 for bad in [block+block,'<thinking>unfinished'+block,'<tool_call>{bad}</tool_call>',block.replace('[499, 898]','[1001, 898]')]:
  try:parse(bad)
  except (ValueError,AssertionError):pass
  else:raise AssertionError(bad)
 assert OLD not in SYSTEM
 return {'tests_passed':True,'thinking_is_not_a_decoder_switch':True}
if __name__=='__main__':print(json.dumps(test()))
