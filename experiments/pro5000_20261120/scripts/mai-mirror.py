from pathlib import Path
p=Path('/data/mai/code/f2f8996/uv.lock')
s=Path('/data/mai/archives/uv.lock.original').read_text()
s=s.replace('https://pypi.org/simple','https://pypi.tuna.tsinghua.edu.cn/simple').replace('https://files.pythonhosted.org/packages','https://pypi.tuna.tsinghua.edu.cn/packages')
p.write_text(s)
