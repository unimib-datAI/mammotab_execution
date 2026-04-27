## README

1. Run the init.sh script to download the mammotab dataset for LLMs
2. Insert the `MODEL_NAME` in the created `.env` file
3. Optional: set `TOKENIZER_NAME` if the tokenizer differs from the model
4. Optional: set `ADAPTER_PATH` to a local PEFT adapter directory containing `adapter_config.json` and `adapter_model.safetensors`; `MODEL_NAME` remains the base model
5. Optional: set `LOAD_IN_4BIT=true` or `LOAD_IN_8BIT=true` if the model does not fit in memory
6. Run docker-compose up
