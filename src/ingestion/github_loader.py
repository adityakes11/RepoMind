import os, shutil, stat
from git import Repo
from langsmith import traceable
IGNORE_DIRS={".git",".venv","venv","node_modules","__pycache__",".idea",".vscode","dist","build"}
SUPPORTED_EXTENSIONS={".py",".js",".jsx",".ts",".tsx",".java",".dart",".cpp",".c",".h",".html",".css",".md",".json",".yaml",".yml",".txt"}

def _remove_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)

@traceable(run_type="chain", name="CloneRepository")
def clone_repository(github_url,repo_name):
    base=os.path.join("data","repos"); path=os.path.join(base,repo_name); os.makedirs(base,exist_ok=True)
    if os.path.exists(path): shutil.rmtree(path,onerror=_remove_readonly)
    Repo.clone_from(github_url,path); return path
@traceable(run_type="chain", name="GetRepositoryFiles")
def get_repository_files(repo_path):
    out=[]
    for root,dirs,names in os.walk(repo_path):
        dirs[:]=[d for d in dirs if d not in IGNORE_DIRS]
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS: out.append(os.path.join(root,name))
    return out
