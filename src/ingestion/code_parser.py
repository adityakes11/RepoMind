import os
from langsmith import traceable

@traceable(run_type="chain", name="ReadSourceFile")
def read_file(path):
    try:
        with open(path,"r",encoding="utf-8") as f:return f.read()
    except (UnicodeDecodeError,OSError):return ""
@traceable(run_type="chain", name="DetectSourceLanguage")
def get_language(path):
    return {".py":"python",".js":"javascript",".jsx":"javascript",".ts":"typescript",".tsx":"typescript",".java":"java",".dart":"dart",".cpp":"cpp",".c":"c",".h":"c/cpp",".html":"html",".css":"css",".json":"json",".md":"markdown",".yaml":"yaml",".yml":"yaml",".txt":"text"}.get(os.path.splitext(path)[1].lower(),"unknown")
