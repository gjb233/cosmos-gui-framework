import copy, json, tempfile, unittest
from pathlib import Path
from prepare import inventory, schedule, write_lines
from mixed_batch import select

class PreparationTests(unittest.TestCase):
    def test_pair_by_prompt_not_list_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'snapshot').mkdir()
            def row(image,answer):return {'image':image,'conversations':[
                {'from':'human','value':'Instruction: x'}, {'from':'gpt','value':answer}]}
            for name,rows in [('r.json',[row('a','reason a'),row('b','reason b')]),
                              ('d.json',[row('b','direct b'),row('a','direct a')])]:
                (root/'snapshot'/name).write_text(json.dumps(rows))
            (root/'mixture-counts.json').write_text(json.dumps({'entries':[
                {'source':'x','file':'r.json','variant':'reasoning','samples':2},
                {'source':'x','file':'d.json','variant':'direct','samples':2}]}))
            self.assertEqual(inventory(root,root)['paired_records'],2)
            for line in (root/'inventory.jsonl').read_text().splitlines():
                r=json.loads(line);self.assertTrue(r['direct'].endswith(r['image']))

    def test_targets_do_not_enter_policy(self):
        r={'id':'a','current_image':'now','history':[],'views':{'low':{'instruction':'tap','direct':'click'}},
           'next_image':'later','fm_mask':1}
        d={'id':'a','level':'low','variant':'direct'}
        p,_=select(r,d); changed=copy.deepcopy(r);changed['next_image']='secret'
        changed['views']['low']['direct']='target changed'
        self.assertEqual(p,select(changed,d)[0]);self.assertNotIn('next_image',p)

    def test_reproducible_schedule_and_no_silent_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg={'seed':42,'sources':{'x':{'count':1}},'sampling':{'answer_style':'direct_only'}}
            (root/'c.json').write_text(json.dumps(cfg))
            write_lines(root/'train-manifest.jsonl',[{'id':'x','source':'x','fm_mask':0,
                'views':{k:{'direct':'d','reasoning':'r'} for k in ['low','high']}}])
            schedule(root/'c.json',root,1,1000);a=(root/'schedule-phase1.jsonl').read_bytes()
            schedule(root/'c.json',root,1,1000);self.assertEqual(a,(root/'schedule-phase1.jsonl').read_bytes())
            report=schedule(root/'c.json',root,2,1000);self.assertEqual(report['variant'],{'direct':1000})
            cfg['sources']['missing']={'count':1};(root/'c.json').write_text(json.dumps(cfg))
            with self.assertRaises(AssertionError):schedule(root/'c.json',root,1,2)

if __name__=='__main__':unittest.main()
