![Alt text](image.png)

# RNATransFormer
Code of RNA transformer model. This Transformer can be used to prdict RNA 3D structure if you have an RNA sequence. Model gives output as 3D point x,y,z for each neucliotide from the RNA sequence.


### Model Weights intialization
#### Initially before training the model on data. Weights initialization will work as following
1. Linear Layer's : PyTorch uses a method called Kaiming uniform initialization by default
2. Embeddings     : For the nn.Embedding layer, PyTorch initializes the weights from a uniform distribution.
3. LayerNorm : nn.LayerNorm layers initialize their weights to 1 and biases to 0 by default.

### Key changes in the original model architecture.
#### 1. Model decoder layer :
- Original RibonanzaNet has decoder output dimensions as 2 cause it was used to predict the reactivity of the neucleiotides.
- Updating it to 3 cause we are training the model for 3D RNA structure predicition. Our new model will output 3 logits for x,y,z respectively for 3D dimensions.

#### 2. Outer product Mean module :
- Original outer product mean module was implemented using einstein equation of matrix multiplication.
- We have updated it to do the same using simple metric multiplications.
