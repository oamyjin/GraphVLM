from vllm import LLM, SamplingParams  # Replace `llm_library` with the actual library you're using

# Initialize the LLM model
model_path = "/scratch/jl11523/projects/LLaVA/local_model/llava-v1.5-7b"
llm = LLM(model=model_path)

# Input text to summarize
input_text = """
Product Description: Specially designed to provide ...
111414
"""

# Create a prompt for the LLM
prompt = f"Please summarize the following text:\n\n{input_text}"

# Sampling parameters for text generation
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=400
)

# Generate the output
output = llm.generate(prompt, sampling_params)

# Extract the result
result = output[0].outputs[0].text  # Assuming the library uses this structure

# Print the summary
print("Summarized Text:", result)
