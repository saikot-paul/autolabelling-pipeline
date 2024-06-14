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


np.random.seed(8)

class LabelGenerator:
    """
    Class that uses Meta's Llama 3 to generate labels given a clustered dataset
    """

    def __init__(self,
                 ckpt_dir: str,
                 tokenizer_path: str,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_seq_len: int = 512,
                 max_batch_size: int = 10,
                 max_gen_len: Optional[int] = None, 
                 sys_prompt_id: str = None, 
                 output_path: str = None,
                 label_col: str = None
                 ):
        self.ckpt_dir = ckpt_dir
        self.dialogs = []
        self.tokenizer = Tokenizer(tokenizer_path)
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature if temperature is not None else 0
        self.top_p = top_p if top_p is not None else 0.5
        self.max_seq_len = max_seq_len if max_seq_len is not None else 512
        self.max_batch_size = max_batch_size if max_batch_size is not None else 10
        self.output_path = output_path if output_path is not None else 'output.csv'
        self.label_col = label_col if label_col is not None else 'Label'
        self.max_gen_len = max_gen_len
        self.output_df = pd.DataFrame()
        self.content_len = len(self.tokenizer.encode(
            (prompt_p1 + prompt_p2), bos=True, eos=True))
        self.cluster_text = []
        self.cluster_text_length = []
        self.cluster_text_items = []
        self.sys_prompt_id = sys_prompt_id

    @staticmethod
    def time_function(func):
        """
        Purpose: 
            - Static function used to time how long a method takes to run
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"Function '{func.__name__}' executed")
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"\t- args: {args}")
            print(f"\t- kwargs: {kwargs}")
            print(f"\t- Time Elapsed: {end_time - start_time:.4f} seconds")
            return result
        return wrapper

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
    
    @time_function
    def read_file(self, file_path, drop_na_col=None):
        """
        Purpose: 
            - Function used to read the intermediate file or dataframe that has already been clustered, sets this dataframe as a class variable 
        
        Args: 
            - file_path: path to which the clustered dataset exists 
            - drop_na_col: column to drop should there be any NaN, nan, na values 
            
        Returns: 
            - dataframe: dataframe that is read 
        """
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

        if drop_na_col is not None:
            self.df = self.df.dropna(subset=drop_na_col)

        return self.df

    def extract_title(self, text):
        """
        Purpose: 
            - Function that uses a regex to extract label from the LLM response 
        
        Args: 
            - text: LLM response 
        
        Returns: 
            - title: extracted title from the response to the prompt 
        """
        
        pattern = re.compile(r': \"(.*?)\"|: (.*?)(?=\n|$)')
        match = pattern.search(text)
        
        if match:
            return match.group(1)
        else:
            return text
     
    def format_str(self, string): 
        fstring = re.sub('\s+',' ', string)
        fstring = fstring.strip()
        words = fstring.split()

        for i in range(len(words)): 
            if i % 20 == 0: 
                words.insert(i, '<br />')

        return ' '.join(words)

    @time_function
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
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size
        )
        return self.generator

    def get_text_list(self, df, text_col='text'):
        """
        Purpose: 
            - Function that aggregates text from the dataframe passed to it. Iterates through the dataframe and appends till the 
        max sequence len is reached 
        
        Args: 
            - dataframe: subset to extract text from 
            - text_col: optional parameter, used to specify which column to extract text from 
            
        Returns: 
            - string: one big blob of text that has been aggregated from the subset 
        """
        
        total_tokens = self.content_len
        limit = math.floor(self.max_seq_len * 0.75)
        temp_text = []

        for _, row in df.iterrows():
            if row['num_tokens'] + total_tokens > limit:
                break
            temp_text.append(row[text_col])
            total_tokens += row['num_tokens']
            
        return "***".join(temp_text) 
    
    def get_stopwords(self, path='stopwords.txt'): 
        """
        Purpose: 
            - Getter function for stopwords specified at path
        
        Args: 
            - path: path to stopwords file 
        
        Return: 
            - stopwords: list of all stopwords 
        """
        
        with open(path, 'r') as f: 
            self.stopwords = f.read().split()
        
        return self.stopwords 
    
    def get_word_list(self, sentence, stopwords=None): 
        """
        Purpose: 
            - Gets a list/array of words that are not in the stopwords list 
            
        Args: 
            - sentence: string that is to be rid of stopwords 
            
        Returns: 
            - word_list: list of words that are not in stopwords list 
        """
        if stopwords is None: 
            stopwords = self.stopwords
            
        word_list = [word.lower() for word in sentence.split() if word not in stopwords and word.isalpha()]
        
        return word_list 
    
    def process_word_group(self, subset, k=10, stopwords=None): 
        
        if stopwords is None: 
            stopwords = self.stopwords 
        
        counter = Counter() 
        
        for sentence in subset: 
            words = self.get_word_list(sentence) 
            counter.update(words)
        
        top_k_words = [word for word, count in counter.most_common(k)]
        
        return top_k_words
    
    @time_function
    def prepare_text(self, text_col='text', sdedup=False, sampling_col=None, random=True):
        """
        Purpose: 
            - Function used to generate an array of text where each element represents the aggregation of text with respect to an individual cluster. 
            - Iterate through each cluster
                - Check if sampling_col is specified: 
                    - Yes = perform stratified sampling 
                        - Check if sampling_col has more than 1 unique entity 
                            - Yes = perform stratified sampling 
                            - No = perform random sampling 
                   - No = perform random 
            - Extracts list of top 10 most common words in each cluster 
        
        Args: 
            - text_col: specify which column text exists in 
            - sampling_col: if you want to do stratified sampling, use this column to specify which column is used to equally distribute samples from, meaning the text_col will have
            equal distribution from unique items from the sampling column 
        
        Returns: 
            - None 
            - Sets class variables: 
                - self.cluster_text: an array of text where each element is an aggregation with respect to a cluster
        """
        
        if not text_col: 
            text_col = 'text'
        
        self.df['length'] = self.df[text_col].apply(lambda x: len(str(x).strip()))
        self.df['num_tokens'] = self.df[text_col].apply(lambda x: len(self.tokenizer.encode(str(x), eos=True, bos=True)))
        df = self.df[self.df['length'] >= 0]
        
        if sdedup: 
            df = df.drop('Drop')
        
        num_clusts = df['clust_id'].nunique() - 1 
        
        
        for i in range(-1, num_clusts): 
            subset = df[df['clust_id'] == i] 
            
            if not random:
                tmp = subset.head(40)
                
            elif sampling_col is not None: 
                
                sample_entities = df[sampling_col].nunique()
                
                if sample_entities > 1: 
                    tmp = self.stratified_sampling(subset, sampling_col, text_col)
                else: 
                    tmp = self.regular_sampling(subset, text_col)
            else: 
                tmp = self.regular_sampling(subset, text_col)
            
            tmp_text = self.get_text_list(tmp, text_col) 

            self.cluster_text.append(tmp_text)   
        
        

    def stratified_sampling(self, subset, sampling_col, text_col='text'): 
        """
        Purpose: 
            - Allow for stratified sampling given a subset of data 
        
        Args: 
            - subset: subset of the original data, representative of a cluster 
            - sampling_col: column that is used as a filter for equal distribution of text 
            - text_col: column to aggregrate text from 
        
        Returns: 
            - sample_groups: dataframe that has equal distribution/samples based on the unique values existing in the sampling_col
        
        """
        
        counts = subset[sampling_col].value_counts()
        min_sample = counts.min()
        sample_groups = subset.groupby(sampling_col)[subset.columns].apply(lambda x: x.sample(n=min_sample)).sample(frac=1).reset_index(drop=True)
        sample_groups = sample_groups.dropna(subset=text_col)
        return sample_groups 

    def regular_sampling(self, subset, text_col='text'):
        """
        Purpose: 
             - Function that randomly samples the subset passed  
        
        Args: 
            - subset: subset of the original dataframe, representative of cluster 
            - text_col: column that specifies where text is located 
            
        Returns: 
            - sample_groups: randomly shuffled subset 
        
        """
        sample_groups = subset.sample(frac=1).reset_index(drop=True)
        sample_groups = sample_groups.dropna(subset=text_col)
        return sample_groups 

    @time_function
    def generate_labels(self, text_col='text'):
        """
        Purpose: 
            - Function used to generate labels 
            - Creates a dataframe: 
                - Columns: 
                    - Cluster: specify the cluster the label belongs to 
                    - Text: aggregated text used to generate the label 
                    - Label: label generated by the LLM
            - Maps the labels to the input file used (clustered dataset) 
        
        Args: 
           - Uses class variables: 
               - self.cluster_text: 
                   - aggregation of text with respect to a cluster 
        Returns: 
            - None 
        """
        
        self.build_generator()
        labels = []
        
        if (self.sys_prompt_id == 1): 
            sys_prompt = self.system(sys_p1)
        
        if (self.sys_prompt_id == 2): 
            sys_prompt = self.system(sys_p2)
            
        if (self.sys_prompt_id == 3): 
            sys_prompt = self.system(sys_p3)
            
        for i, t in enumerate(self.cluster_text):
            user_prompt = self.user(prompt_p1 + t + prompt_p2)
            self.dialogs.append([sys_prompt, user_prompt])

        for i, item in enumerate(self.dialogs):
            results = self.generator.chat_completion(
                [item],
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            print(results)
            tmp = results[0]['generation']['content']
            labels.append(tmp)
            print(f'>Assistant: {tmp}')

        num_clusts = range(-1, self.df['clust_id'].nunique()-1)
        text = [t.strip() for t in self.cluster_text]
        counts = self.df.clust_id.value_counts().sort_index()
        
        label_col = self.label_col
        
        self.output_df = pd.DataFrame({
            'Cluster': num_clusts,
            'Text': text,
            label_col: labels, 
            'Cluster Count': counts 
        })
        
        self.output_df[label_col] = self.output_df[label_col].apply(
            lambda x: self.extract_title(x))
        self.output_df['Text'] = self.output_df['Text'].str.strip()
        
        label_mapping = dict(zip(num_clusts, labels))
        self.df[label_col] = self.df['clust_id'].map(label_mapping)
        self.df[label_col] = self.df[label_col].apply(
            lambda x: self.extract_title(x))
        
        self.df['format_text'] = self.df['text'].apply(lambda x: self.format_str(str(x)))
    
    @time_function
    def generate_labels_cove(self, text_col='text'):
        """
        Purpose: 
            - Function used to generate labels 
            - Creates a dataframe: 
                - Columns: 
                    - Cluster: specify the cluster the label belongs to 
                    - Text: aggregated text used to generate the label 
                    - Label: label generated by the LLM
            - Maps the labels to the input file used (clustered dataset) 
            - Implements chain of 
        
        Args: 
           - Uses class variables: 
               - self.cluster_text: 
                   - aggregation of text with respect to a cluster 
        Returns: 
            - None 
        """
        
        self.build_generator()
        labels = []
        
        if (self.sys_prompt_id == 1): 
            sys_prompt = self.system(sys_p1)
        
        if (self.sys_prompt_id == 2): 
            sys_prompt = self.system(sys_p2)
            
        if (self.sys_prompt_id == 3): 
            sys_prompt = self.system(sys_p3)
            
        for i, t in enumerate(self.cluster_text):
            user_prompt = self.user(prompt_p1 + t + prompt_p2)
            self.dialogs.append([sys_prompt, user_prompt])

        for i, item in enumerate(self.dialogs):
            
            print('======================================================================================')
            
            results = self.generate_chat(item)
            item.append(results[0]['generation'])
            
            print(f' Representational Text Truncated: {self.cluster_text[i][:300]}\n')
            for q in questions: 
                user_q = self.user(q) 
                item.append(user_q) 
                print(f'>User: {q}\n')
                results = self.generate_chat(item) 
                content = results[0]['generation']['content']
                item.append(results[0]['generation'])
                print(f'>Assistant: {content}\n')
                
            tmp = results[0]['generation']['content']
            labels.append(tmp)

        num_clusts = range(-1, self.df['clust_id'].nunique()-1)
        text = [t.strip() for t in self.cluster_text]
        counts = self.df.clust_id.value_counts().sort_index()
        
        label_col = self.label_col
        
        self.output_df = pd.DataFrame({
            'Cluster': num_clusts,
            'Text': text,
            label_col: labels, 
            'Cluster Count': counts 
        })
        
        self.output_df[label_col] = self.output_df[label_col].apply(
            lambda x: self.extract_title(x))
        self.output_df['Text'] = self.output_df['Text'].str.strip()
        
        label_mapping = dict(zip(num_clusts, labels))
        self.df[label_col] = self.df['clust_id'].map(label_mapping)
        self.df[label_col] = self.df[label_col].apply(
            lambda x: self.extract_title(x))
        
        self.df['format_text'] = self.df['text'].apply(lambda x: self.format_str(str(x)))
        
    
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
        
    @time_function
    def generate_output(self): 
        """
        Purpose: 
            - Generate output csvs of the clustered dataset with labels for each entry and just the labels 
        
        Args: 
            - None 
        
        Returns: 
            - None
            - Output files are created based on the parameters used to initialize the class
        """
        
        self.output_df.to_csv(self.output_path, index=False)
        self.df.to_csv(self.file_path, index=False) 
        print(f'\t - Saved output to: {self.output_path}')
        
