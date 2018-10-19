# Open AI Balance
Good model for OpenAI's cart pole environment

## Approach
- Obtaining the environment:
  - Using OpenAI's Gym library to initialize the 'cart pole' environment

- Creating the train dataset:
  - Run 100.000 episodes with random actions
  - Select the observations only from actions that have high scores *score>75* (aprox 37.000 obs to learn from)
  - The average exploration score was *in my case* 81.

- Training the model:
  - For shaping the model I used Keras
  - The model contains:
    - 5 hidden layers (each one is using the rectifier activation)
      1. 32 neurons
      2. 64 neurons
      3. 128 neurons
      4. 256 neurons
      5. 256 neurons
    - 5 dropouts (for each layer)
    - softmax output
    
- Getting results:
  - Run 100 episodes and get the average score for all (*in my case* I got an almost doubled test score - 148)
  
## The saved model
- The saved model in the 'model' folder gets the following metrics:
  - 148.62 average score after training
  - 61.09% accuracy for the training set on 12 epochs
  
![alt text](https://i.imgur.com/koDA96d.gif)
### Happy balancing :)
