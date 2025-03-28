# Hands_On_LLM_WR
This repository is create to test code and concepts present in the book Hand On LLM by Jay Alammar, Maarten Grootendorst

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
7. 
