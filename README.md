# Hands_On_LLM_WR
This repository is create to test code and concepts present in the book Hand On LLM by Jay Alammar, Maarten Grootendorst

### Ch-1: An Introduction to Large Language Models

1. In this book, **we learn how large language models (LLMs) have revolutionized the field of Language AI**, significantly altering our methods for tasks like translation, classification, and summarization. This trend, driven by rapid advancements in deep learning, has enabled language AI systems to achieve unprecedented levels of text understanding and generation.
2. **We begin by establishing that Language AI** is a field within artificial intelligence focused on enabling computer systems to perform tasks that closely resemble human intellectual capabilities, such as speech recognition and language translation.
3. **We trace the history of Language AI**, noting its progression from earlier models aimed at structured language representation for computers, commencing with techniques like **bag-of-words**.
4. **We explain that the bag-of-words technique** involves tokenizing text into individual words or subwords and subsequently creating numerical representations (vectors) by counting the occurrences of these tokens. We categorize these as **representation models**.
5. **We then introduce embeddings** as a more advanced form of numeric representation for words, where the goal is to capture semantic similarity. These embeddings allow us to measure how conceptually close the meanings of different words are using distance metrics.
6. **We describe the attention mechanism** as a significant step towards enabling models to encode context within sequences. This allows a model to focus on the relevant parts of an input and amplify their signal, selectively determining the importance of words within a sentence. We further elaborate on **self-attention**, a key component in Transformer models, which allows simultaneous attention to different positions within a single sequence.
7. **We highlight two main categories of LLMs discussed in this book:** representation models (encoder-only) like **BERT**, which utilize **masked language modeling** during training, and generative models (decoder-only) like the **GPT family**. Both of these categories leverage the power of the attention mechanism.
8. **We emphasize that LLMs have numerous common use cases and applications**, and within this book, we will explore their practical application in areas such as copywriting and summarization, the creation of semantic search systems, and their utility for text classification, search, and clustering.
9. **We state that this chapter provides an overview** of the broader landscape of Language AI, encompassing its diverse applications, the important societal and ethical implications that arise from its use, and a consideration of the resources typically required to operate these models.
10. **We conclude this introductory chapter** by demonstrating the generation of our first text using **Phi-3-mini**, a relatively small yet performant generative model that we will utilize in practical examples throughout the remainder of the book.

### Ch-2: Tokens and Embeddings

1. In Chapter 2, **the book delves into the core concepts of tokens and embeddings**, which are fundamental for understanding large language models (LLMs).
2. **We begin by exploring what tokens are and the different tokenization methods** that LLMs utilize. The book highlights that current interactions with language models primarily involve generating text one token at a time.
3. **The book illustrates how tokenizers break down text into words or parts of words** based on specific methods and training procedures.
4. **The book points out that three main factors dictate how a tokenizer processes an input prompt**: the **tokenization method** (like BPE or WordPiece), **tokenizer design choices** (vocabulary size, special tokens), and the **training dataset**.
5. **The book discusses various tokenization schemes**, including **subword tokens** (the most common), **word tokens**, **character tokens**, and **byte tokens**, outlining their differences and capabilities in handling new or rare words.  
6. **It showcases how different trained LLM tokenizers**—such as BERT, GPT-2, GPT-4, StarCoder2, FLAN-T5, Galactica, Phi-3, and Llama 2—process the same input text, emphasizing their unique behaviors and the use of **special tokens** like `[CLS]`, `[SEP]`, and `<|endoftext|>`.  
7. **The book clarifies that embeddings are numeric representations used to capture meaning and patterns in language**. LLMs operate on raw, static embeddings as inputs and generate **contextual text embeddings** as outputs.  
8. **It dives into the influential word2vec embedding method**, a foundational approach that preceded modern LLMs, explaining how it learns word relationships and encodes them as vectors.  
9. **The book emphasizes the usefulness of embeddings for measuring semantic similarity**, often leveraging distance metrics to compare meanings between words.  
10. **It also covers various types of embeddings**, including **word embeddings** and **sentence/text embeddings**, which reflect different abstraction levels and power diverse applications.  
11. **The book explains the word2vec algorithm’s training process**, which involves predicting surrounding words (or neighbors) and using negative sampling to refine the embeddings.  
12. **It reiterates that modern language models generate high-quality, contextualized token embeddings**, which are essential for downstream tasks like named-entity recognition, summarization, and classification.  
13. **The book concludes by reinforcing that tokenizer algorithms, tokenization parameters, and training datasets are core design decisions** that shape how tokenization behaves.

### Ch-3: Looking inside Large Language Model

1. In this chapter we learn the inner workings of a transformer-based LLM. We look at some of the main intuitions of how transformer models work.
2. For a text generation model, the model does not generate the text all in one operation, it actually generates one token at a time. Each token generation step is one forward pass through the model. After each token generation, we tweak the input prompt for the next generation step by appending the output token at the end of the input prompt. These kind models which consume their earlier prediction to generate later predictions are called auto-regressive models
3. Forward Pass - Components of forward pass include tokenizer, transformer blocks, and language modeling head (LM head). The tokenizer is followed by the neural network: a stack of Transformer blocks that do all of the processing. That stack is then followed by the LM head, which translates the output of the stack into probability scores for what the most likely next token is.
4. The method of choosing a sin‐ gle token from the probability distribution is called the decoding strategy. The idea here is to basically sample from the probability distribution based on the probability score. Choosing the highest scoring token every time is called greedy decoding. It’s what happens if you set the temperature parameter to zero in an LLM.
5. For text generation, only the output result of the last stream is used to predict the next token. When generating a token, we simply append the output of previous pass and do another forward pass. In a transformer model, we do not need to repeat calculations of the previous streams. This optimizations technique technique called the keys and values (kv) cache and it provides a significant speedup of the generation process. Keys and values are some of the central components of the attention mechanism.
6. A Transformer block is made up of two successive components:
    * The attention layer is mainly concerned with incorporating relevant information from other input tokens and positions
    * The feedforward layer houses the majority of the model’s processing capacity
7. Two main steps are involved in the attention mechanism:
    i. A way to score how relevant each of the previous input tokens are to the current token being processed (in the pink arrow).
    ii. Using those scores, we combine the information from the various positions into a single output vector.
8. To give the Transformer more extensive attention capability, the attention mechanism is duplicated and executed multiple times in parallel.
9. While calculating attention, the goal is to produce a new representation of the current position that incorporates relevant information from the previous tokens
10. The training process produces 3 projection matrices - **query projection matrix**, **key projection matrix**, and **value projection matrix**.
11. The two steps of attention are -
   i. Relevance Scoring - In this step, the query vector of the current position is multiplied by keys matrix. This produces a score stating how relevant each of the previous token is. These scores are then normalized by passing through a softmax function
   ii. Combining Information - In this step we multiple the value vector associated with each token by that token's score. Summing up those vectors produces results of this step
12. Recent improvement to transformers architecture include -
   * Local/Sparse attention which limits the context of previous tokens that the model can attend to
   * Multi-query and grouped-query attention presents a more efficient attention mechanism by sharing the keys and values matrices across multiple/all the attention heads.
   * Flash attention - a method to speed up training and inference by speeding up the attention calculation by optimizing what values are loaded and moved between a GPU’s shared memory (SRAM) and high bandwidth memory (HBM).
   * Other improvment to architectures include using residual connection, improvements in normalization (RMSNorm), and activation functions (e.g. - SwiGLU)
   * Improvment to positional embeddings include packing, rotary embeddings.



### Ch-4: Text Classification
1. Text Classification can be done with representation model or generative models
2. We perform sentiment analysis on rotten tomatoes movie reviews (Positive/Negative::1/0)
3. We use frozen representation model for our classification task. The representation model we use are foundation models. A foundation model is fine-tuned for specific tasks; for instance, to perform classification or generate general-purpose embeddings.
4. We explore two types of representation models, a task specific model and an embedding model
5. For task specific model - we chose the Twitter-RoBERTa-base for Senti‐ment Analysis model and obtained an f1-score of 0.8
6. For classification using embedding models, we performed it in two ways -
    1. using embedding from embedding model and train a logistic classifier. The embedding model is frozen
    2. We also explored performing classification using cosin similarity where we claculated similarity each review in text set against average of positive and negative class embeddings in the training set and assigned a class based on similarity. This way is generally helpful if we do not have classes
7. Next, we explored way to perform text classification using generative models. These models take as input some text and generative text and are thereby aptly named sequence-to-sequence models.
8. For generative models, we used
    1. Text-to-Text transformer, we used an open source encoder-decoder Flan-T5 model
    2. ChatGPT for classification. This is decoder only model
  
### Ch-5: Text Clustering and Topic Modeling
#### Text Clustering
1. In this chapter, we will first explored how to perform clustering with embedding models and then transitioned to a text-clustering-inspired method of topic modeling, namely BERTopic.
2. We ran clustering and topic modeling on ArXiv articles
3. We explored a common pipeline for text clustering
   1. Convert documents to embeddings with embedding model, in this example we used 'thenlper/gte-small' model
   2. reduce the dimentionality of embeddings with dimentionality reduction model. We can use PCA or UMAP but UMAP tends to handle non-linear relationships better
   3. Find groups of semantically similar documents with a cluster model. we use HDBSCAN as it does not force a data point to be a part of the cluster and can also find outliers 
#### Topic Modeling
1. Classic approaches, like latent Dirichlet allocation, assume that each topic is charac‐ terized by a probability distribution of words in a corpus’s vocabulary.
2. Topic Modeling is used to find themes or latent topics in a collection of textual data
3. BERTopic is modular i.e. you can apply different building blocks to different step in topic model to customize your own topic model based on your own needs
4. The different components for topic modeling using BERTopic are -
    1. Calculate embeddings for document - 
    2. Perform dimentionality reduction - UMAP
    3. Cluster the compressed embeddings - HDBSCAN
    4. To create topic representation
       1. Create a class of bag-of-words
       2. Weigh terms - TF-IDF
5. Each of the components can be changed
6. Once the topics have been assigned, -1 is the topic contains all documents that could not be fitted within a topic and are considered outliers.
7. Representation Model. We can use special techniques/ models to better represent the topics. some techniques are - 
    1. KeyBERTInspired - KeyBERT extracts keywords from texts by comparing word and document embeddings through cosine similarity.
    2. Marginal Maximal Relevance (MMR) - We can use maximal marginal relevance (MMR) to diversify our topic representa‐ tions. The algorithm attempts to find a set of keywords that are diverse from one another but still relate to the documents they are compared to.
    3. Generative Models - We can use generative models to give the representation words and exmaple documents and ask it to output a topic name
      * Flan T5
      * ChatGPT 3.5
  
### Ch-6: Prompt Engineering
1. Through Prompt Engeering we can design prompts in such a way that enhances the quality of output
2. In this chapter we use an open source model - 'Phi-3 Mini'
3. Model output can be controlled using parameters-
    * temperature: Temperature controls the randomness or creativity of the text generated. Higher temperature increases the likelihood that less probable tokens are generated and vice versa.
    * top_p: Top_p is know as nucleus sampling. It will consider tokens until it reaches cummulative probability
    * top_k: Top_k parameter control how many token an LLM can consider to generate next token
#### Intro to Prompt Engineering
4. The main objective of prompt engineering is to elicit a use response from the model. The main ingredients of a good prompt are - instruction and the data it refers to
5. Some more technique used to improve the quality of LLM output are -
    * Specificity: accurately describe what you want to achieve
    * Hallucination: ask LLM to only generate the answer if it knows the answer
    * Order: Either begin or end your prompt with instruction as LLMs tend to focus more on beginning of the prompt (primacy effect) or end of the prompt (recency effect)
#### Advance Prompt Engineering
6. A good prompt contains the some or all of the following components, we can explore building up the context slowly
    * Persona: Describe what role you want the LLM to take. For e.g. - you are an expert clinician
    * Instruction: The task itself, you want to be as specific as possible. For e.g. - create a technical document notes based on discuss of an algorithm
    * Context: It answers the question - "What is the reason for this instruction?" For e.g - you are a data scientist working with clinical experts to come up with a heuristic approach to solve a domain problem
    * Format: The format of the output that the LLM should generate. For e.g. - create a one-pager technical document
    * Audience: The target of the generated text. For e.g. - Create a document which will be read by clinical experts and technical leadership
    * Tone: Tone of the text to be generated. For e.g. - technical document language should be formal
    * Data: Data for the task. For e.g. - provide the details of technical document, these could be free-hand meeting notes or brainstorming session notes
7. In context learning: We can provide LLM with examples of exactly the things that we want to achieve. Its called Zero-shot, One-shot, Few-shot learning based on how many examples.
8. Chain Prompting: Instead of giving all the information at once, we can break them down between prompts, this technique allows LLM to spend more time on one question instead of tackling whole problem at once.
9. Chain of thought: It aims to have the generative model think first rather than answering the question directly without any reason (Sytem 2 thinking). We can use this by asking the model to provide the reasoning behind the answer. E.g.- Add the prompt 'Let's think step-by-step'
10. Self-Consistency: This method asks the generative model the same prompt multiple times and takes the majority result as the final answer. It require the same question to be asked multiple times
11. Tree of thought: Exploring Intermediate Steps: By leveraging a tree-based structure, generative models can generate inter‐ mediate thoughts to be rated. The most promising thoughts are kept and the lowest are pruned.
12. Output Verification: Reasons for validating output might include - getting a structured output, generate a valid output (e.g. - classification), ethics, accuracy. There are 3 ways of controlling output of LLM - examples, grammar, and fine-tuning.
13. llama-cpp-python is a library like transformers which expect compressed (quantized) model (GGUF format)

### Ch-7: Advanced Text Generation Techniques and Tools
1. In this chapter, we explore methods and concepts for improving the quality of generated text - Model I/O, Memory, Agents, Chains. These methods are all integrated within Langchain framework.
2. In this chapter we use a GGUF model, a GGUF model represents a compressed version of its original counterpart through a method called quantization, which reduces the number of bits needed to represent the parameters of an LLM. Quantization is a similar process that reduces the precision of a value (e.g., removing seconds) without removing vital information
3. Instead of copy-pasting this template each time we use the LLM in LangChain, we can use one of LangChain’s core functionalities, namely “chains.”  With LangChain, we will use chains to create and use a default prompt template.
4. We create a template for prompt which is expected by the LLM, using this template, the model takes in a system_prompt, which generally describes what we expect from the LLM. Adding a prompt template to the chain is just the very first step you need to enhance the capabilities of your LLM.
5. With **sequential chains**, the output of a prompt is used as the input for the next prompt.
6. In this chapter, we generate a story by providing a summary. We generate the title, charater description as intermediate steps and chain all the steps together
7. When we are using LLMs out of the box, they will not remember what was being said in a conversation. To make these models stateful, we can add specific types of memory to the chain we created earlier, common method to help LLM remember conversations are:
   * Conversation Buffer - We can add conversation buffer by adding entire conversation (conversation buffer) or last k steps of the conversation (windowed conversation buffer). This can be achieved by LangChain’s ConversationBufferMemory and ConversationBufferWindowMemory  respectively.
   * Conversation Summary - Instead of passing the conversation history directly to the prompt, we use another LLM to summarize it first.
8. Agents are a powerful technique in which we can enable LLMs to perform actions by providing it tools which it can execute
9. Although the tools they use are important, the driving force of many agent-based systems is the use of a framework called Reasoning and Acting (ReAct). ReAct merges these two concepts and allows reasoning to affect acting and actions to affect reasoning. In practice, the framework consists of iteratively following these three steps: thought, action, observation.
10. The LLM is asked to create a “thought” about the input prompt. This is similar to asking the LLM what it thinks it should do next and why. Then, based on the thought, an “action” is triggered. The action is generally an external tool, like a calculator or a search engine. Finally, after the results of the “action” are returned to the LLM it “observes” the output, which is often a summary of whatever result it retrieved.
11. We see an example where the LLM uses DuckDuckGo's search tool to run a query online and calculator tool to perform mathematical operation before outputting a results.


### Ch-8: Semantic Search and Retrieval-Augmented Generation (RAG)
1. Semantic search enables searching by meaning of text, not just keyword search. RAG systems are one of the most popular way to reduce hallucinations, we provide the LLM the block of text which it should use to generate text.
2. The way to use language models for search are -
   * Dense Retreival: retreives the nearest neighbour after converting query and text to embeddings
   * Reranking: Uses a reranking model to rank the items obtained using dense retreival based on user-defined criteria
   * RAG: retrieves the text which is most similar to the query and augments the LLM with the information in order to provide it better context
3. Dense Retreival - We chunk the document before proceeding to embed each chunk, these chunks are stores in vector database and are ready for retreival
4. We build an example for dense retreival using Cohere to search the Wikipedia page for the film Interstellar. In this example, we build a search index using `chromadb` library on the embeddings. We also look at keyword search using the `BM250kapi` library
5. Caveats of using a dense search include -
   * If the text dont contain the the answer, dense retreival still gets results and distances.
   * Exact matches are are hard to find, that is a case for keyword search.
   * Dense retreival systems find it challenging to work properly in domains other than the ones they have been trained on.
   * Determining best way to chunk documents
6. Transformer language models are limited in context size so intelligent chunking strategies should be applied in order to retreive most relevent documents. Generally multiple vectors per document perform better than one vector per document strategies due to their ability to capture more information
7. To make the search scale beyond millions of vectors, we use nearest neighbour optimized search libraries like FAISS or Annoy. Vector databases like Pinecone, ChromaDB, Weaviate can also be used for building retrieval systems
8. Fine-tuning a model based on the data include making the relevant queries closer to the document and making irrelevant queries farther from the document
9. Rerankers operate as a part of a search pipeline with a goal to reorder a number of shortlisted search results by relevance. The shortlisting step is called the first-stage and can be a dense retrieval, keyword search or hybrid approach
10. A reranker assigns a relevance score to each document by looking at the document and the query at the same time. One popular way to build LLM search reranker is to present the query and each results to an LLM working as a cross-encoder
11. MAP (Mean Average Precision) and nDCG (normalized discounted cummulative gain) are two metrics to evaluate search systems. In this book, we learn how to calculate MAP for a search system.
12. The mean average precision takes into consideration the average precision score of a system for every query in the test suite. By averaging them, it produces a single metric that we can use to compare a search system against another.
13.  A basic RAG pipeline is made up of a search step followed by a grounded generation step where the LLM is prompted with the question and the information retrieved from the search step.
14.  We turn a search system to RAG system by adding a generation step at the end an LLM of the search pipeline. The generation step is called grounded generation
15.  RAG systems also help the users by citing its sources
16.  In this book we build example of RAG system using LLM API and local model
17.  A prompt template plays a vital part in the RAG pipeline. It is the central place where we communicate the relevant documents to the LLM.
18.  There are several techniques to improve performance of a RAG system
   *   Query Rewriting: Precise query writing
   *   Multi-query RAG: multiple queries to get the answer
   *   Multi-hop query: series of sequential queries
   *   Query rerouting: using multiple sources
   *   Agentic RAG
19. RAGs are evaluated on the basis of - fluency, perceived utility, citation recall, citation precision.
20. We can use LLM-as-a-judge and score the LLM across various criterion. `Ragas` is a software that does this.

### Ch-9: Multimodal Large Language Models

1. Both the original Transformer as well as the Vision Transformer take unstructured data, convert it to numerical representations, and finally use that for tasks like classification.
2. ViT involves a method for tokenizing images into “words,” which allowed them to use the original encoder structure. It converts the original image into patches of images.
3. ViT converts the orignial image into patches of images, these patches are linearly embedded to create numerical representations, namely embeddings. These can be used as input to transformer model. The moment the embeddings are passed to the encoder, they are treated as if they were textual tokens.
4. Multimodel embedding models create the embeddings for multiple modalities in the same vector space. The pair of embeddings are compared using cosine similarity.
5. When we start training, the similarity between the image embedding and text embed‐ ding will be low as they are not yet optimized to be within the same vector space. During training, we optimize for the similarity between the embeddings and want to maximize them for similar image/caption pairs and minimize them for dissimilar image/caption pairs. This method is called contrastive learning. Eventually, we expect the embedding of an image of a cat would be similar to the embedding of the phrase “a picture of a cat”. Negative examples of images and captions should also be included in the training.
6. BLIP-2 is a modular technique which allows for introducing vision capabilities to existing language models. Instead of building the architecture from scratch, BLIP-2 bridges the vision-language gap by building a bridge, named the Querying Transformer (Q-Former), that con‐ nects a pretrained image encoder and a pretrained LLM. The Querying Transformer is the bridge between vision (ViT) and text (LLM) that is the only trainable component of the pipeline.
7. To connect the two pretrained models, Q-Former mimics their architecture, it has 2 modules - An image transformer to interact with Vision Transformer for feature extraction and text transformer that can interact with the LLM
8. In step 1, image-document pairs are used to train the Q-Former to represent both images and text. 
9. The images are fed to the frozen ViT to extract vision embeddings. These embeddings are used as the input of Q-Former’s ViT. The captions are used as the input of Q-Former’s Text Transformer. With these inputs, the Q-Former is then trained on three tasks:
   * Image-text contrastive learning - This task attempts to align pairs of image and text embeddings such that they maximize their mutual information.
   * Image-text matching - A classification task to predict whether an image and text pair is positive (matched) or negative (unmatched).
   * Image-grounded text generation - Trains the model to generate text based on information extracted from the input image.
These three objectives are jointly optimized to improve the visual representations that are extracted from the frozen ViT. I
10. In step 2, the learnable embeddings derived from step 1 now contain visual informa‐ tion in the same dimensional space as the corresponding textual information. There is also a fully connected linear layer in between them to make sure that the learnable embeddings have the same shape as the LLM expects.
11. We looks at how the BLIP model preprocesses text and image and then look at two usecases - Image captioning and Multi-modelchat based prompting.

### Ch-10: Creating Text Embedding Models
1. In this chapter, the book discusses a variety of ways to create and fine-tune an embedding model to increase its representative and semantic power.
2. An embedding model can be trained for a variety of purposes, for e.g.- for training a sentiment classifier, we can fine-tune the model such that documents are closer in n-dimensional space based on their sentiment rather than their semantic nature.
3. Contrastive Learning - Contrastive learning is a technique that aims to train an embedding model such that similar documents are closer in vector space while dissimilar documents are further apart.
4. Before sentence-transformers, sentence embeddings often used an archi‐ tectural structure called cross-encoders with BERT. A cross-encoder allows two sentences to be passed to the Transformer network simultaneously to predict the extent to which the two sentences are similar.
5. sentence-transformers are trained using bi-encoders, bi-encoder or SBERT for sentence- BERT. Although a bi-encoder is quite fast and creates accurate sentence representa‐ tions, cross-encoders generally achieve better performance than a bi-encoder but do not generate embeddings.
6. Steps to train bi-encoders
   * In sentence-transformers the classification head is dropped, and instead mean pooling is used on the final output layer to generate an embedding. This pooling layer averages the word embeddings and gives back a fixed dimensional output vector.
   * The training for sentence-transformers uses a Siamese architecture. In this architecture, we have two identical BERT models that share the same weights and neural architecture. These models are fed the sentences from which embeddings are generated through the pooling of token embeddings. Then, models are optimized through the similarity of the sentence embeddings.
   * During training, the embeddings for each sentence are concatenated together with the difference between the embeddings. Then, this resulting embedding is optimized through a softmax classifier.
7. 
