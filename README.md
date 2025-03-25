# Harmonica
Music Generation GenAI project
- **Dataset on Kaggle**
  - https://kaggle.com/datasets/927554b78ed40565b8e0bb96ec442b85100ffb286ae2f741bd4f434080b0438e
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
