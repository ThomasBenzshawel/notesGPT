# Ratrieval Augmented Generative Model
https://research.ibm.com/blog/retrieval-augmented-generation-RAG

The project goal is to create an RAG model that can have highly flexible domain knowledge.

All intensive computations are done on MSOE's HPC cluster "ROSIE" https://www.msoe.edu/about-msoe/news/details/meet-rosie/

Any non-intensive computations are done on my at-home Unraid server with an Nvidia Tesla T4

# THIS GUIDE IS AMAZING FOR UNDERSTANDING MULTI GPU IN PYTORCH

https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# Timeline

- [x] Understand transformers
- [x] Crate a transformer network that can be trained across multiple GPUs! 
- [ ] Read "Attention is all you need"
- [x] Understand GPT models
- [ ] Develop a way to replicate the "chat" experience that comes with chat-gpt
- [ ] Use a "semantic search engine" to find files that contain info related to a search (similar to google's page rank system)
- [ ] Combine a fine tuned GPT model to a specific data set with a semantic search engine
- [ ] Test the results
