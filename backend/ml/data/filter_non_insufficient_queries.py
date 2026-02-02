import json

in_path = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/generated_queries_train.jsonl'
out_path = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/generated_queries_train_non_insufficient.jsonl'

count = 0
with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        query = data.get('generated_query')
        if query and str(query).strip().lower() != 'insufficient_info':
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1
print(f'완료: {out_path}, {count}개 row 저장됨')