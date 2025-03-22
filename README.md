# Harmonica
Music Generation GenAI project
- **Dataset on Kaggle**
  - https://kaggle.com/datasets/bd5662249d974c929233629a3651eedfe35efbe32b89138bc9b9a547f03c8aae
  - https://kaggle.com/datasets/efff01786fd1e03abc4fbf286b843e4194a805b061c8efe716afe522ebbf2de9
- **Pre Processing**
  - pre-processing midi files into json format
  - converting the json file to muisc tokens
  - tag the music tokens with emotions like happy,sad,neutral,calm and energetic
  - then convert these tagged music tokens to tagged remi tokens
- **Prompt Engineering**
  - Chain of Thought (CoT)
  - Tree of Thought (ToT)
  - Graph of Thought (GoT)
- **Music RAG**
  -  use the csv tagged music tokens as the context for retrieval
  -  use an Agent(LLAMA model from Huggingface) to convert music data to NLP
  -  Langchain FAISS vector store implementation
  -  facebook musicgen model as generator
  -  implement prompt engineering techniques

