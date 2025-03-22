# Harmonica
Music Generation GenAI project
- **Dataset on Kaggle**
  - https://kaggle.com/datasets/f6e95217392b2f95d53d199d824066e1efcd1b9a99fe1e11528adbf4d53f80d8
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

