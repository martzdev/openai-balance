#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:44:56 2018

@author: mihaisturza
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import gym

def create_model():
    model = Sequential()
    
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(2,activation="softmax"))
    
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    
    return model

def create_env():
    env = gym.make('CartPole-v0')
    return env

def create_dataset(env):
    X_train, y_train, scores = [], [], []
    print("Starting exploration")
    for _ in range(100000):
        obs = env.reset()
        X_samples, y_samples, score = [], [], 0
        for step in range(10000):
            action = np.random.randint(0,2)
            hot_action = np.zeros(2)
            hot_action[action] = 1
            X_samples.append(obs)
            y_samples.append(hot_action)
            obs, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > 75:
            scores.append(score)
            X_train += X_samples
            y_train += y_samples
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    avg_score = np.mean(scores)
    
    print("Average exploration score: {}".format(avg_score))
    
    return X_train, y_train, avg_score

def train_model(model, env):
    X_train, y_train, exp_score = create_dataset(env)
    model.fit(X_train, y_train, epochs = 12)
    
    scores = []
    
    for _ in range(100):
        obs = env.reset()
        score = 0
        for step in range(5000):
            env.render()
            action = np.argmax(model.predict(obs.reshape(1,4)))
            obs, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > 85:
            scores.append(score)
    
    print("Average score: {}".format(np.mean(scores)))
    print("The model is {} times better than in exploration phase.".format(np.mean(scores)/exp_score))
    
    return model
    
def save_model(model):
    with open("./model/structure.json","w") as file:
        model_json = model.to_json()
        file.write(model_json)
        print("Saved model structure")
    model.save_weights("./model/weights.h5")
    print("Saved model weights")
    
def main():
    env = create_env()
    model = create_model()
    trained_model = train_model(model, env)
    save_model(trained_model)
    
if __name__ == "__main__":
    main()