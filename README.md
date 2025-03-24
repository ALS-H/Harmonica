# Harmonica
Music Generation GenAI project
- **Dataset on Kaggle**
  - https://kaggle.com/datasets/a9eab5da230bf7f0d338625fc7f837a757296e82ec8e8c93e2bff753a4ceada3
- **Pre Processing**
  - pre-processing midi files into json format
  - converting the json file to muisc tokens csv
  - tag the music tokens with emotions like happy,sad,neutral,angry,fear,calm,disgust,energetic and surprise.
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
