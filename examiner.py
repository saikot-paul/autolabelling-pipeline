import fire
import math
import re
import time
import numpy as np 
import pandas as pd
from prompt import *
from llama import Dialog, Tokenizer, Llama
from typing import List, Optional
from functools import wraps


class Examiner: 
    
    def __init__(
        self, 
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_seq_len: int = 256,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None, 
        output_path: str = None,
    ): 
        
        self.ckpt_dir = ckpt_dir
        self.tokenizer = Tokenizer(tokenizer_path)
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature if temperature is not None else 0
        self.top_p = top_p if top_p is not None else 0.5
        self.max_seq_len = max_seq_len if max_seq_len is not None else 512
        self.max_batch_size = max_batch_size if max_batch_size is not None else 2
        self.output_path = output_path if output_path is not None else 'output.csv'
        self.max_gen_len = max_gen_len
        self.build_generator() 

        
    
    def user(self, content: str) -> dict:
        """
        Purpose: 
            - Function that returns a dictionary representing a user prompt
        
        Args: 
            - content: prompt/item you want to ask Meta 
        
        Returns: 
            - dictionary for the dialogue 
        """
        
        return {'role': 'user', 'content': content}

    def system(self, content: str) -> dict:
        """
        Purpose: 
            - Function that is used to generate a system prompt 
        
        Args: 
            - content: prompt/item you want to ask Meta 
        
        Returns: 
            - dictionary for the dialogue 
        """
        
        return {'role': 'system', 'content': content}
    
    def assistant(self, content: str) -> dict:
        """
        Purpose: 
            - Function that is used to generate a system prompt 
        
        Args: 
            - content: prompt/item you want to ask Meta 
        
        Returns: 
            - dictionary for the dialogue 
        """
        
        return {'role': 'assistant', 'content': content}

    
    def build_generator(self):
        """
        Purpose: 
            - Function used to create the generator (LLM) 
        
        Args: 
            - self: uses the values passed to the file when called in terminal to instantiate a generator 
            
        Returns: 
            - generator
        """
        
        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=7000,
            max_batch_size=self.max_batch_size
        )
        return self.generator
    
    def generate_chat(self, item):
        """
        Purpose: 
            - Given dialogue items, LLama will respond to the latest prompt 
        
        Args: 
            - item (list[Dialogue]): list of dictionaries that contain either system/user/assistant prompt 
        
        Returns: 
            - results (list[Dialogue]): list of dictionary containing LLM response to latest prompt
        """
        
        results = self.generator.chat_completion(
            [item], 
            max_gen_len=self.max_gen_len, 
            temperature = self.temperature, 
            top_p = self.top_p
        ) 
        
        return results
    
    def read_file(self, path): 
        """
        Purpose: 
            - Read csv file 
        
        Arguments: 
            - path (str): path to csv file 

        Returns: 
            - None: sets the csv as instance dataframe (self.df)
        """
        self.df = pd.read_csv(path)
    
    def process_text(self, text_col='text', label_col='Label'):
        """
        Purpose: 
            - Process text from dataframe, create list of tuples (x,y)
                - x: text used to label cluster 
                - y: label used to generate label 
        
        - Arguments: 
            - text_col (str): column to extract representational text 
            - label_col (str): column to extract label from 

        - Returns 
            - text_and_labels list[tuple]: list of tuples of representational text and labels
        """
        text = self.df[text_col].to_list()
        label = self.df[label_col].to_list()

        text_and_labels = zip(text, label)
        return text_and_labels
        
        
    def examine_labels(self, txt, lbl): 
        """
        Purpose: 
            - Function that evaluates the labels generated 

        Arguments: 
            - text_col (str): 
        """        

        print('=========================================================================')

        prompt_p1 = f'Below is an exerpt of a cluster where the text was used to generate a label.\nText: "{txt}"'
        prompt_p2 = f'\nHere is the label created for the text. Label: "{lbl}"'
        prompt_p3 = """
        Given the text and the label, evaluate the label on the following criteria: \n\t1. Ability to accurately capture recurring topics, messages, and themes.\n\t2. Ability to accurately capture main perspectives and narratives.\n\t3. General ability to capture the essence of the text.\nKeep answers concise but also explain what is done well and what could be added. 
        """
        
        og_q = self.user(prompt_p1)
        og_a = self.assistant(prompt_p2)
        prompt = self.user(prompt_p3)
        

        results = self.generate_chat([og_q, og_a, prompt])
        
        print(f'>User:\n{prompt_p1}')
        print(f'>Assistant:\n{prompt_p2}')
        print(f'>User:\n{prompt_p3}')
        print(f">Assistant:\n{results[0]['generation']['content']}")


"""
def main(path: str, label: str): 
            
    examiner = Examiner(ckpt_dir='Meta-Llama-3-8B-Instruct/', tokenizer_path='Meta-Llama-3-8B-Instruct/tokenizer.model')

    df = pd.read_csv(path)

    txt_lbl = zip(df['Text'].to_list(), df[label].to_list())

    for txt, lbl in txt_lbl: 
        examiner.examine_labels(txt, lbl)
        
if __name__ == '__main__': 
    fire.Fire(main)
"""