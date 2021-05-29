#!/usr/bin/env python
# coding: utf-8


import sys
import subprocess
import os


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print (package)


def downloadDialoGPT(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.save_pretrained(path)
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.save_pretrained(path)


def downloadDialoGRPT(path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-human-vs-rand")
    tokenizer.save_pretrained(path)
    
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-human-vs-rand")
    model.save_pretrained(path)


def checkModel():
    DialoGPTPath=os.path.join(os.getcwd(),r"DialoGPT-medium")
    if (not os.path.exists(DialoGPTPath)) or (len(os.listdir(DialoGPTPath) ) == 0):
        downloadDialoGPT(DialoGPTPath)
    else:
        print('DialoGPT exist')

    pathRPTUpdown = os.path.join(os.getcwd(),"DialogRPT-human-vs-rand")
    if (not os.path.exists(pathRPTUpdown)) or (len(os.listdir(pathRPTUpdown)) == 0):
        downloadDialoGRPT(pathRPTUpdown)
    else:
        print('DialogRPT-human-vs-rand exist')


if __name__=="__main__":

    packages = ['nltk==3.6.2',
                'transformers==4.6.0',
                'torch',
                'Flask_SocketIO==5.0.3',
                'numpy==1.18.5',
                'Flask==1.1.2',
                'tensorflow==2.1.0']

    for package in packages :
        install(package)
        
    checkModel()
