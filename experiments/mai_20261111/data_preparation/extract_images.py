"""Safely extract multipart tar.gz archives without producing another full tar copy."""
import argparse, json, shutil, subprocess, tarfile
from pathlib import Path

def extract(root):
    src=root/'snapshot/data/images'; dest=root/'extracted';dest.mkdir(exist_ok=True)
    for first in sorted(src.glob('*.tar.gz.part-000')):
        name=first.name.rsplit('.part-',1)[0]; parts=sorted(src.glob(name+'.part-*'))
        assert [int(p.name.rsplit('-',1)[1]) for p in parts]==list(range(len(parts)))
        marker=dest/(name+'.complete.json')
        if marker.exists(): continue
        process=subprocess.Popen(['cat',*[str(p) for p in parts]],stdout=subprocess.PIPE)
        count=0
        try:
            with tarfile.open(fileobj=process.stdout,mode='r|gz') as archive:
                for member in archive:
                    target=(dest/member.name).resolve()
                    assert target.is_relative_to(dest.resolve()), 'Unsafe archive path'
                    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                    assert member.isfile(), 'Links and special files are not accepted'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_name(target.name+'.extracting')
                    with archive.extractfile(member) as source,temporary.open('wb') as out:
                        shutil.copyfileobj(source,out)
                    assert temporary.stat().st_size==member.size
                    temporary.replace(target);count+=1
            process.stdout.close();assert process.wait()==0
        except BaseException:
            process.terminate();process.wait();raise
        marker.write_text(json.dumps({'archive':name,'files':count,'parts':[p.name for p in parts]}))
        print(name,count,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);extract(p.parse_args().root)
