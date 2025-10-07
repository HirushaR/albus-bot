# 🤖 Albus Bot  
A simple **AI Chatbot** built with **Python**, powered by **Neural Networks** and **Deep Learning** using **TensorFlow**, **TFLearn**, and **NLTK**.

---

## 🧠 Overview  
Albus Bot is a conversational chatbot capable of understanding and responding to user input using natural language processing (NLP).  
It was designed as a foundational project to explore how **neural networks** and **intent classification** can be applied to create human-like conversation systems.

---

## 🚀 Features  
- Built using **TensorFlow** and **TFLearn** for neural network training  
- Utilizes **NLTK** for text preprocessing and tokenization  
- Employs **NumPy** for numerical computations  
- Trains on custom **intents.json** file for conversational context  
- Capable of classifying user input into predefined intents  
- Modular and easily extendable for adding new responses or intents  

---

## 🧩 Tech Stack  
- **Python 3.x**  
- **TensorFlow**  
- **TFLearn**  
- **NLTK**  
- **NumPy**  
- **JSON**

---


## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HirushaR/albus-bot.git
   cd albus-bot
1. **Install dependencies:**
   ```bash
   pip install tensorflow tflearn nltk numpy
1. **Train the model:**
   ```bash
   python train.py
1. **Run the chatbot:**
   ```bash
   python chat.py

---
## 🗣️Usage
- Modify the intents.json file to define new patterns and responses.
- Run the chatbot and start chatting!
- The model will classify the user's intent and generate contextually relevant replies.

---
## 📚 Example intents.json
  ```bash 
  {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello!", "Hi, how can I help you today?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you", "Goodnight"],
      "responses": ["Goodbye!", "Talk to you later!"]
    }
  ]
}


  
