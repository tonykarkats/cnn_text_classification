# Initialization
Run ./create_embeddings to obtain the .bin file containing the python dictionary giving the embeddings for the words in the training set, for the dimensions given in the script (same as the Stanford glove embeddings).

# Use
After that, the model is available as a dictionary, use as follows:
```python
model = wv.load('test-phrases.bin')
dog = model['dog']
```

Here, dog is a vector of size the number of features (given as a parameter in wv.py).
