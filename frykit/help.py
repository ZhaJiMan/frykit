from pathlib import Path
import shutil

def new_dir(dirpath):
    '''新建目录.'''
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

def del_dir(dirpath):
    '''删除目录. 目录不存在时会报错.'''
    dirpath = Path(dirpath)
    shutil.rmtree(str(dirpath))

def renew_dir(dirpath):
    '''重建目录.'''
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir(parents=True)