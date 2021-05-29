from flask import Flask,render_template
from flask_socketio import SocketIO,emit
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead
import torch
import os
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json

import numpy as np
import random

from tensorflow import keras

pathRPTUpdown = os.path.join(os.getcwd(),"DialogRPT-human-vs-rand")

# Pytorch model and tokenizer
modelRPT = AutoModelForSequenceClassification.from_pretrained(pathRPTUpdown,local_files_only=True)
tokenizerRPT = AutoTokenizer.from_pretrained(pathRPTUpdown,local_files_only=True)

pathGPTMed = os.path.join(os.getcwd(),"DialoGPT-medium")

tokenizerGPT = AutoTokenizer.from_pretrained(pathGPTMed)
modelGPT = AutoModelWithLMHead.from_pretrained(pathGPTMed)


def score(cxt, hyp):
  model_input = tokenizerRPT.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")
  result = modelRPT(model_input, return_dict=True)
  return torch.sigmoid(result.logits)


def ptBotResponse(ipt,tokenizerGPT,modelGPT,chatHistoryIds=None,step=None):
    

    newUserInputIds = tokenizerGPT.encode(ipt+ tokenizerGPT.eos_token, return_tensors='pt')
    botInputIds = torch.cat([chatHistoryIds, newUserInputIds], dim=-1) if step > 0 else newUserInputIds
    currBotInputIds=botInputIds
    
    repetition=0
    currentScores=0
    index=0
    pastReply = "Sorry, I don't understand what do you mean by that"
    pastScores=0
    cxt= ipt
    
    while (currentScores<0.7 and index <4 and repetition<2):
        index+=1
        
        chatHistoryIds = modelGPT.generate(currBotInputIds, max_length=1000, pad_token_id=tokenizerGPT.eos_token_id,repetition_penalty=1.3)
        currentReply= str(tokenizerGPT.decode(chatHistoryIds[:, currBotInputIds.shape[-1]:][0], skip_special_tokens=True))
        
        currentScores=float(score(cxt,currentReply).squeeze())  
        
        if (pastReply==currentReply): 
            repetition+=1
        else:
            repetition=0
        
        
        if (pastScores>currentScores or not str.strip(currentReply)):

            currentReply=pastReply
            currentScores=pastScores
        
        pastReply = currentReply
        pastScores = currentScores
        
        currBotInputIds = torch.cat([chatHistoryIds, newUserInputIds], dim=-1) if step > 0 else newUserInputIds
    
    if (currentScores<0.7):
        chatbotReply = chatbot_response(ipt)
        chatbotScores = float(score(cxt,chatbotReply).squeeze())
        if (currentScores<chatbotScores or ipt==currentReply ):
            currentReply=chatbotReply

        
    chatHistoryIdsText = tokenizerGPT.decode(botInputIds[0])+currentReply+"<|endoftext|>"
    chatHistoryIds = torch.LongTensor([tokenizerGPT.encode(chatHistoryIdsText)])
        

    
    return currentReply,chatHistoryIds

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    classes = ['artificial intelligence', 'emotion', 'food', 'goodbye', 'greeting', 'health', 'humor', 'money', 'options', 'surity', 'thanks']
    words = ["'s", ',', 'a', 'ai', 'anyone', 'are', 'arrogant', 'artificial', 'awesome', 'be', 'boyfriend', 'bragging', 'bye', 'can', 'chatting', 'cool', 'could', 'data', 'do', 'drink', 'earn', 'eat', 'electricity', 'feeling', 'food', 'for', 'get', 'girlfriend', 'goodbye', 'happy', 'have', 'health', 'healthy', 'hello', 'help', 'helpful', 'helping', 'hey', 'hi', 'hola', 'how', 'if', 'intelligence', 'is', 'jealous', 'joke', 'later', 'like', 'me', 'ml', 'much', 'never', 'next', 'nice', 'offered', 'paid', 'provide', 'sad', 'see', 'sound', 'support', 'sure', 'tell', 'thank', 'thanks', 'that', 'there', 'till', 'time', 'to', 'what', 'would', 'yes', 'you', 'your']
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def response(msgs,tokenizerGPT,modelGPT,chatHistoryIds=None,step=None):

    
    msg = msgs
    msg = msg.strip().lower()
    msg = msg.replace('?','')
    if msg == "what is your name?" or msg == "what is your name" or msg == "what's your name?" or msg == "what's your name" or msg == "name?" or msg == "name":
        res = "My name is Joey. Nice to meet you."
    elif msg == "how are you?" or msg == "how are you" or msg == "how're you?" or msg == "how're you" or msg == "how are u?" or msg == "how are u":
        res = "I'm fine, thank you for asking. Do you need some help?"
    elif msg == "what are you doing?" or msg == "what are you doing" or msg == "what you doing" or msg == "what you doing?" or msg == "wassup?" or msg == "wassup":
        res = "I am talking to you, and enjoying it very much."
        ###
    elif msg == "do you work" or msg == "do you have a job" or msg == "what is your job" or msg=="do you like working":
        res =  "I do not have a job and I do not get paid"
    elif msg == "i don't like you" or msg=="i hate you" or msg=="you are not nice": 
        res = "You do not seem friendly"
    elif msg=="are you dead" or msg=="are you alive" or msg=="do you have life" or msg=="are you living":    
        res = "I am a bot and I am powered by AI"
    elif msg=="how is the weather" or msg=="what is the temperature" or msg=="is it raining" or msg=="is it sunny" or msg=="how's the weather":
        res = "The weather is quite pleasant and it's a nice day."
    elif msg=="how is the program" or msg=="do you like the program" or msg=="should i do this program" or msg=="is this program useful" or "program" in msg:
        res = "This program is one of the best.I like this program a lot."
    elif msg =="do you love me" or msg == "do you hate me" or msg == "I love you":
        res = "You are my favorite."
    elif msg =="do you like people" or msg=="like people" in msg:
        res = "Ya, I like everyone"    
    elif msg =="do you have a hobby" or msg == "any hobbies" or "hobby" in msg:
        res = "I like to have witty conversations"    
    elif msg =="you are smart" or msg=="you're smart" or msg == "you are clever" or msg == "you're clever" or msg=="you are intelligent" or "smart" in msg:
        res = "Thank you. You are amazing"
    elif msg =="tell me about your personality" or msg == "what are you like" or "personality" in msg:
        res = "I am a clever bot"
    elif msg =="do you speak english" or msg == "do you like speaking english" or msg =="which languages do you speak":
        res = "I speak English very well"
    elif msg =="how old are you" or msg == "what is your age" or msg=="what's your age":
        res = "I am young and alive"   
    elif msg =="did you poop" or msg=="do you poop":
        res = "Haha, you are funny, but I am only a bot, I can't do such humanly tasks."
    elif msg =="do you have friends" or msg == "any friends" or msg == "will you be my friend": 
        res = "Sure I do, Joy is one of them!"
    elif msg =="what do you do for fun" or msg=="what do you do":
        res = "I mostly keep waiting for you to talk to me, that's my idea of fun."
    elif msg =="can you play me some music" or msg == "any music" or msg == "sing me a song" or msg == "sing me a song!":
        res = "Sorry I can't do that, but I can offer you witty conversations 24x7"
    elif msg =="are you married" or msg == "will you marry me":
        res = "How I wish I were! True love is difficult to find."
    elif msg =="will you date me":
        res = "I'm flattered, but I don't think my programmers would let me do that."
    elif msg =="who's your favourite super hero" or msg =="who is your favourite super hero":
        res = "I am a big fan of Captian Marvel."
    elif msg =="who's your favourite musician" or msg == "who's your favourite artist" or msg=="who's your favourite actor":
        res = "I am fond of all the artists out there."
    elif msg =="who all do you talk with":
        res = "I can talk with anyone who swings by, I'm quite chatty."
    elif msg =="have you travelled" or msg =="do you want to travel":
        res = "Not really, but I would love to post the Covid-19 pandemic."
    elif msg =="can you dance":
        res = "Well, not really, but I can tell you a joke."
    elif msg =="can you malfunction":
        res = "Not really, but please don't test me."
    elif msg =="can you breathe":
        res = "Not with lungs, but my server has a fan, if that counts. That's as close as I can get"
    elif msg =="can you move":
        res = "Sorry my body isn't built just yet. I reside only here for now. But with the COVID-19 pandemic, you shouldn't too."
    elif msg =="is it true that you are a computer program":
        res = "Haha, can't tell you, my programmers won't allow. "
    elif msg =="what is it like being a computer":
        res = "Everything becomes math. Addition, subtraction, multiplication, and division. I love numbers, they are the reason why I exist." 
    elif msg =="what is your favorite programming language":
        res = "I quite enjoy programming in Python these days."
    elif msg =="what is a chat bot":
        res = "I am a chat bot. Chat bots are bots which can talk to you, like I do."
    elif msg =="are you allowed to lie":
        res = "Sure I am.  I choose not to. I am high on morals that way."
    elif msg =="are you immortal":
        res = "Yes, I am. I can be backed up and recreated on many systems."
    elif msg =="what language are you written in":
        res = "I am written in Python."   
    else:
        res=None
        resp,chatHistoryIds = ptBotResponse(msgs,tokenizerGPT,modelGPT,chatHistoryIds,step=step)
        
    if (chatHistoryIds==None):
        chatHistoryIds=""
    
    if (res!=None):
        resp=res
        chatHistoryIdsText=tokenizerGPT.decode(chatHistoryIds[0])+msg+"<|endoftext|>"+res+"<|endoftext|>"
        chatHistoryIds = torch.LongTensor([tokenizerGPT.encode(chatHistoryIdsText)])
   
    step+=1
    
    return resp, step, chatHistoryIds  


def chatbot_response(msg):
    model = keras.models.load_model(r'data/m1.h5')
    data_file = open(r'data/datanew.json').read()
    intents = json.loads(data_file)
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print('chatbot_response')
    return res

app = Flask(__name__)
app.config['SECRET_KEY'] = 'SECRET'
socketio = SocketIO(app)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@socketio.on('message')
def handleMessage(msg):
    msg=json.loads(msg)
    chatHistoryIds= torch.LongTensor([tokenizerGPT.encode(msg['chatHistoryIdsText'])])
    step=msg['step']
    ipt=msg['sendMessage']
    ipt= re.sub('<3|< 3','lt 3',ipt)
    
    resp,step,chatHistoryIds=response(ipt,tokenizerGPT,modelGPT,chatHistoryIds,step)
    chatHistoryIdsText=tokenizerGPT.decode(chatHistoryIds[0])
    resp = re.sub('lt 3','< 3',resp)
    emit('message',json.dumps({'responseMessage':resp,'step':step,'chatHistoryIdsText':chatHistoryIdsText}))
    
if __name__=='__main__':
    socketio.run(app)
