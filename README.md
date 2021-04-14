## 1. Author of this exercice 2
- Mathieu Tardy
- Jérémie Feron
- Tom Huix
- Benjamin Bonvalet

## 2. Description of our algorithm


Data preprocessing:
We treat the problem as sentence pair classification, where we feed the algorithms two sentences separated by the SEP token : 
- The first one corresponds to the review sentence
- The second one is a concatenation with '#' character of the aspect category and the words to review (for example seating, food, etc...)

We pass this newly obtained string through BERT's tokenizer to transform the word tokens into IDs.

Model:
First layer (feature extraction): We use a BERT model pre-trained and finetuned on cross domain reviews dataset
(data from Amazon and Yelp) for aspect based sentiment analysis(Xu et .al, 2019). Given the small size of the dataset
we don't fine tune the model and simply use it as a feature extractor. 
Note: We extract only the embedding of the classificationtoken [CLS] (the first token for each sentence-pair) and feed it to the next layer.

Next, we define and train a simple feed forward neural network that takes as input the embedding of the [CLS] token.
The network architecture is rather simple with one hidden layer and dropout followed by a softmax (BertAsc class in model.py) to classify between the 3 sentiments : positive, negative and neutral. 

For the loss, we used a Cross Entropy Loss and an AdamW optimiser from transformers library in 
order to train our model.

As hyperparameters, we choose a batch size of 16, a max len of the sentence of 120 and 3 
epochs as recommended in the Bert paper (Attention is all you need).


## 3. Results on the dev dataset :
With all these hyperparameters, we have an accuracy of 87.23% on the dev dataset.


## Reference of the model's paper:

@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = "jun",
    year = "2019",
}
