cp .env.example .env
fileId="1f02LvZtpYomTKN8E6xVQTNy8AGURkAms"
curl -L "https://drive.usercontent.google.com/download?id=${fileId}&export=download&confirm=t" -o "work/mammotab_sample.jsonl"
