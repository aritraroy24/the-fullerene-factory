import torch
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig

model_name_or_id = "OpenDFM/ChemDFM-v1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")

#input_text = """I need 6 COOH groups on my addend and for it to have some planar aromatic structure. i want to add four of these addends to a C60. i don’t mind what reaction is used to attach the addends to the fullerene. Generate SMILES of the three most viable addend candidates, with synthesis routes and expected structures."""
input_text = """Here’s an addend, OH, i want to add multiple times to C84 (isomeric mixture). show me all the possible structures i might expect for a tetrakis-adduct."""

input_text = f"[Round 0]\nHuman: {input_text}\nAssistant:"

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    do_sample=True,
    top_k=20,
    top_p=0.9,
    temperature=0.9,
    max_new_tokens=1024,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
print(generated_text.strip())
