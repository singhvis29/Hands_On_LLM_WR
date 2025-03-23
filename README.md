# Hands_On_LLM_WR
This repository is create to test code and concepts present in the book Hand On LLM by Jay Alammar, Maarten Grootendorst

Ch - 4: Text Classification
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
