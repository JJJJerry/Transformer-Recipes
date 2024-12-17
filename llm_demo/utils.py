import json
def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(fp=f)
    return data
def load_jsonl(path):
    res=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            data=json.loads(line)
            res.append(data)
    return res
def save_json(data,path):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(obj=data,fp=f,ensure_ascii=False)
def save_jsonl(data:list,path):
    with open(path,'w',encoding='utf-8') as f:
        str_list=[]
        for d in data:
            s=json.dumps(obj=d,ensure_ascii=False)
            str_list.append(s+'\n')
        f.writelines(str_list)