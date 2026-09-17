import json,sys
from pathlib import Path
from src.rag.retriever import retrieve_documents
def evaluate(collection_name):
    with open(Path(__file__).with_name("dataset.json"),encoding="utf-8") as f:data=json.load(f)
    correct=0
    for item in data:
        docs=retrieve_documents(item["question"],collection_name,5); got={d.metadata.get("file") for d in docs}
        if set(item["expected_files"])&got:correct+=1;print("✅",item["question"])
        else:print("❌",item["question"],"Expected:",item["expected_files"],"Retrieved:",got)
    print(f"\nRetrieval Recall: {correct/len(data):.2%}" if data else "No evaluation data")
if __name__=="__main__":
    if len(sys.argv)!=2: print("Usage: python -m evaluations.evaluate <collection_name>");sys.exit(1)
    evaluate(sys.argv[1])
