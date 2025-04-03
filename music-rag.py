!pip install langchain langchain-community
!pip install faiss-cpu

import faiss
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore

# Load DataFrame
df = pd.read_csv("/kaggle/input/tagged-music-tokens/tagged_music_data.csv")  # Adjust path

# Initialize Hugging Face Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Get embedding dimension
embedding_dim = len(embeddings.embed_query(" "))

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)

# Create FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# ✅ Process in Batches to Reduce Memory Usage
batch_size = 500  # Adjust based on available memory
music_data = df.to_dict(orient="records")  # Convert rows to dictionaries
texts = [str(row) for row in music_data]  # Convert each row to a string

for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]  # Process batch-wise
    vector_store.add_texts(batch)
    print(f"✅ Processed {i + batch_size} records")

print("✅ FAISS Vector Store Ready!")



from transformers import pipeline

# 🔹 Load LLaMA 1B Model Correctly
HF_TOKEN = "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"
llama_pipeline = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B", 
    device_map="auto", 
    token=HF_TOKEN  # ✅ Correct way to pass token
)


from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to('cuda')


# 🚀 **Agent: Convert Music Data to Text Representation**
def generate_music_prompt(user_input):
    """
    Uses FAISS to retrieve similar music tokens and converts them into a natural language music description.
    """
    # 🔍 Retrieve similar stored music descriptions
    retrieved_docs = vector_store.similarity_search(user_input, k=5)

    # ✅ Extract text properly
    retrieved_texts = [doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")]

    if not retrieved_texts:
        return "No relevant music data found."

    # 🎼 Format prompt for LLaMA
    prompt = (
        "You are a music expert that converts structured music token data (pitch, duration, velocity, emotion) "
        "into a natural language description. Convert the following retrieved data into a meaningful text prompt "
        "for a text-to-music model:\n\n"
        f"{retrieved_texts}\n"
        "Describe the music in terms of mood, tempo, and instruments used."
    )

    # 📝 Generate response using LLaMA
    response = llama_pipeline(prompt, max_length=256)[0]["generated_text"]
    return response  # 🎶 Return the formatted text prompt




#Chain of Thought

def generate_music_prompt(user_input):
    """
    Uses FAISS to retrieve similar music tokens and applies Chain of Thought (CoT) reasoning
    to generate a detailed and structured music prompt.
    """
    # 🔍 **Step 1: Retrieve Relevant Musical Data**
    retrieved_docs = vector_store.similarity_search(user_input, k=3)
    retrieved_texts = [doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")]

    if not retrieved_texts:
        return "No relevant music data found."

    # 🧠 **Step 2: Reasoning about the Music Style**
    reasoning_steps = (
        "**Step 1: Identify Mood and Emotion**\n"
        "- Extract the mood (e.g., calm, energetic, melancholic) from the retrieved data.\n\n"
        "**Step 2: Identify Tempo and Dynamics**\n"
        "- Determine if the tempo is slow, moderate, or fast.\n"
        "- Identify dynamics like soft (piano) or loud (forte).\n\n"
        "**Step 3: Identify Instrumentation**\n"
        "- Determine which instruments dominate (e.g., piano, violin, guitar).\n\n"
        "**Step 4: Structure a Final Music Prompt**\n"
        "- Construct a natural language music description incorporating the above features.\n"
    )

    # 📝 **Step 3: Format Prompt for LLaMA**
    prompt = (
        "You are a music expert that converts structured music token data (pitch, duration, velocity, emotion) "
        "into a natural language description.\n\n"
        "Follow these steps:\n"
        f"{reasoning_steps}\n"
        "Now, based on the following retrieved data, generate a structured and meaningful text prompt for a text-to-music model:\n\n"
        f"{retrieved_texts}\n"
        "Describe the music in terms of mood, tempo, and instruments used."
    )

    # 🎼 **Step 4: Generate Structured Response using LLaMA**
    response = llama_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]

    return response  # 🎶 Return the refined music prompt


#Tree of thought

def generate_music_prompt(user_input):
    """
    Uses FAISS to retrieve similar music tokens and applies Tree of Thought (ToT) reasoning
    to generate a detailed and structured music prompt.
    """
    # 🔍 **Step 1: Retrieve Relevant Musical Data**
    retrieved_docs = vector_store.similarity_search(user_input, k=50)
    retrieved_texts = [doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")]

    if not retrieved_texts:
        return "No relevant music data found."

    # 🌲 **Step 2: Tree of Thought Reasoning**
    tree_of_thought = (
        "**Branch 1: Musical Emotion Analysis**\n"
        "- What emotions are commonly associated with the retrieved music tokens?\n"
        "- How do different elements (chords, tempo, dynamics) contribute to the mood?\n\n"

        "**Branch 2: Tempo & Rhythm Structure**\n"
        "- What is the general tempo (BPM range) of the retrieved examples?\n"
        "- Are there noticeable rhythm patterns (e.g., waltz, syncopation)?\n\n"

        "**Branch 3: Instrumentation Breakdown**\n"
        "- What instruments are used in the retrieved samples?\n"
        "- How does instrumentation impact the texture and feel of the music?\n\n"

        "**Branch 4: Genre & Style Refinement**\n"
        "- What genre does this music best fit into?\n"
        "- Are there any unique stylistic elements that stand out?\n\n"

        "**Branch 5: Final Music Prompt Construction**\n"
        "- Combine insights from the previous branches into a well-structured, natural language prompt.\n"
        "- Ensure clarity in describing mood, tempo, instrumentation, and genre."
    )

    # 📝 **Step 3: Format Prompt for LLaMA**
    prompt = (
        "You are a music expert skilled in analyzing structured music tokens (pitch, duration, velocity, emotion) "
        "and transforming them into a rich, human-readable music description.\n\n"
        "Follow this Tree of Thought framework to break down the musical elements before generating a response:\n\n"
        f"{tree_of_thought}\n"
        "Now, using the following retrieved data, generate a well-structured and insightful text prompt:\n\n"
        f"{retrieved_texts}\n"
        "Ensure that the final output describes the music in terms of **emotion, tempo, rhythm, instruments, and style**."
    )

    # 🎼 **Step 4: Generate Structured Response using LLaMA**
    response = llama_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]

    return response  # 🎶 Return the refined music prompt


#Graph of thought

def generate_music_prompt(user_input):
    """
    Uses FAISS to retrieve similar music tokens and applies Graph of Thought (GoT) reasoning
    to generate a detailed and structured music prompt.
    """
    # 🔍 **Step 1: Retrieve Relevant Musical Data**
    retrieved_docs = vector_store.similarity_search(user_input, k=60)
    retrieved_texts = [doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")]

    if not retrieved_texts:
        return "No relevant music data found."

    # 🔗 **Step 2: Graph of Thought Reasoning**
    graph_of_thought = (
        "**Nodes (Core Elements):**\n"
        "- 🎵 Mood & Emotion (happy, sad, energetic, calm, etc.)\n"
        "- 🎼 Tempo & Rhythm (BPM, patterns, time signature)\n"
        "- 🎻 Instrumentation (piano, violin, guitar, etc.)\n"
        "- 🎷 Genre & Style (jazz, classical, electronic, etc.)\n\n"
        
        "**Edges (Interconnections Between Elements):**\n"
        "- How does **tempo** influence the **mood**? (e.g., fast tempo = energetic, slow tempo = relaxing)\n"
        "- How does **instrumentation** shape the **genre**? (e.g., strings = classical, synths = electronic)\n"
        "- How do **rhythm patterns** contribute to **emotion**? (e.g., syncopation in jazz for swing feel)\n"
        "- How do **dynamic variations** affect **style**? (e.g., crescendo for intensity, legato for smoothness)\n\n"

        "**Graph-based Reasoning Steps:**\n"
        "1️⃣ Identify the most relevant nodes (e.g., if the user wants 'relaxing piano melody', focus on **mood**, **tempo**, and **instrumentation**).\n"
        "2️⃣ Find edges (connections) that define relationships (e.g., 'slow tempo' + 'soft piano' → 'calm, meditative atmosphere').\n"
        "3️⃣ Generate a natural language description integrating all relevant nodes and their relationships.\n"
    )

    # 📝 **Step 3: Format Prompt for LLaMA**
    prompt = (
        "You are an expert in music generation who transforms structured music tokens (pitch, duration, velocity, emotion) "
        "into rich, expressive descriptions.\n\n"
        "Use the **Graph of Thought** framework to analyze musical relationships before generating a structured prompt:\n\n"
        f"{graph_of_thought}\n"
        "Now, based on the following retrieved data, generate a well-structured and interconnected music description:\n\n"
        f"{retrieved_texts}\n"
        "Ensure the output describes **mood, tempo, instrumentation, rhythm, and their interconnections**."
    )

    # 🎼 **Step 4: Generate Structured Response using LLaMA**
    response = llama_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]

    return response  # 🎶 Return the refined music prompt


#music generation
from scipy.io.wavfile import write
# 🔹 **Example User Query**
user_query = "Generate an happy piano melody"

# 🎶 Generate refined prompt
final_music_prompt = generate_music_prompt(user_query)
print("Generated Prompt:", final_music_prompt)

# 🎵 **Generate Music using MusicGen**
inputs = processor(text=[final_music_prompt], padding=True, return_tensors="pt").to("cuda")
audio_values = model.generate(**inputs, max_new_tokens=256)

# 🎼 **Convert to WAV and Save**
sample_rate = 16000  # Set sample rate
audio_array = audio_values.cpu().detach().numpy().squeeze()  # Convert tensor to numpy
write("generated_music.wav", sample_rate, audio_array)  # ✅ Now `write` is correctly imported and used

print("✅ Music saved as 'generated_music.wav' 🎶")