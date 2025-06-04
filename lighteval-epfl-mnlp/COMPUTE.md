# CS 552 Modern NLP Project Compute Tutorial

This year, we will be using EPFL's RCP educational cluster section as the main compute for students.
You will be operating on the cluster via the [Gnoto](https://gnoto.epfl.ch/) JupyterLab environment.

## Step 1: Login to Gnoto:
Proceed with the link and login with your **Tequila** credential.

## Step 2: Create Virtual Environment

```bash
my_venvs_create mnlp_m2
my_venvs_activate mnlp_m2
```

### Reconfigure the CC Environment Variable

```bash
#!/bin/bash
# This script wraps the real gcc and adds the -B option.

REAL_GCC="/bin/gcc" # The actual gcc path your Python script is trying to call
GCC_LIBEXEC_DIR="/usr/libexec/gcc/x86_64-linux-gnu/13/"

# Execute the real gcc with the added -B option and all original arguments
exec "$REAL_GCC" -B "$GCC_LIBEXEC_DIR" "$@"
```

Let's put the above into `$HOME/gcc_wrapper`. We export this path to the CC env variable:

```bash
unset GCC_EXEC_PREFIX
export CC="$HOME/gcc_wrapper"
```

## Step 3: Install Default Packages

```bash
pip install torch
pip install transformers
```

## Step 4: Load A Huggingface Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```

## Install evaluation suite

```bash
git clone https://github.com/eric11eca/lighteval-epfl-mnlp.git

cd lighteval-epfl-mnlp

pip install -e .[quantization]
```

## Setup Huggingface Hub Account

Setup an account at the [Huggingface Hub](https://huggingface.co/)