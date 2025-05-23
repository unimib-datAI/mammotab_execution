from generate import LLM

model = LLM(
    model_name="osunlp/TableLlama",
    tokenizer_name="osunlp/TableLlama",
    device="cuda",
    load_in_4bit=False,
    load_in_8bit=False,
)

print(model.generate(["Vacca la puttana"]))