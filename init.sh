cp .env.example .env
fileId="18nTELcDR6RvjIlaNrq6z6BLc8pYGHLzy"
curl -L "https://drive.usercontent.google.com/download?id=${fileId}&export=download&confirm=t" -o "work/mammotab_sample.jsonl"