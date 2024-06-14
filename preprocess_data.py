import re
import numpy as np
import pandas as pd
import time
from cuml.cluster import HDBSCAN, KMeans
from cuml.manifold import UMAP
from functools import wraps
from sentence_transformers import SentenceTransformer

np.random.seed(8)

class DataProcessor:
    """
    Class made to preprocess text data, create embeddings, reduce dimensions and cluster them such that they can be used to generate labels.
    """

    def __init__(self, file_path, embd_model_path):
        """
        Purpose: 
            - Class constructor 
        
        Args: 
            - file_path: path to csv file containing text data
            - embd_model_path: path to the embedding model directory
        
        Returns: 
            - DataProcessor class 
        """
        
        
        self.file_path = file_path
        self.embd_model_path = embd_model_path
        self.data = None
        self.rd = False 
        self.semdup = False
        self.no_url = False
        self.extract_url_col = False
        

    @staticmethod
    def time_function(func):
        """
        Purpose: 
            - Wrapper function to time how long functions take 
        """
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"Function '{func.__name__}' executed")
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"\t- Time Elapsed: {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    def read_file(self, col_to_drop_from=None):
        """
        Purpose: 
            - Read file that was passed in the constructor
            
        Args: 
            - col_to_drop: Drop rows where a column is Nan
        
        Returns: 
            - dataframe (pandas.DataFrame): dataframe of the read file 
        """
        
        if col_to_drop_from is not None: 
            self.df = pd.read_csv(self.file_path).dropna(subset=col_to_drop_from).reset_index(drop=True)
        else: 
            self.df = pd.read_csv(self.file_path)
                    

    def extract_url_helper(self, string):
        """
        Purpose: 
            - Helper function to extract urls from a string 
        
        Args: 
            - string (str): string to parse
        
        Returns: 
            - returns (str) either all matching strings to regex or none
        """
        
        match = re.search(r"https?://\S+", string)
        if match:
            return match.group()
        else:
            return None

    def parse_url_helper(self, string):
        """
        Purpose: 
            - Helper function to extract words from a url
        
        Args: 
            - string (str): string to parse
        
        Returns: 
            - returns string of the words in the url or none 
        """
        
        if string:
            words = string.split('/')
            i = -1

            while abs(i) <= len(words):
                if (words[i] and not re.search('^[0-9]+', words[i])):
                    return ' '.join(words[i].split('-'))
                i -= 1

        return None

    def remove_non_ascii(self, string):
        """
        Purpose: 
            - Helper function to remove non-ascii characters from a string 
        
        Args: 
            - string (str): string to parse
        
        Returns: 
            - returns cleaned string 
        """
        
        return re.sub(r'[^\x00-\x7F]+', '', string)
    
    
    def remove_emojis(self, text):
        """
        Removes emojis from the given text.

        Args:
            text (str): The input text containing emojis.

        Returns:
            str: The text with emojis removed.
        """
        # Emoji pattern
        emoji_pattern = re.compile(
            "["  
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # Chinese characters
            "\U00002702-\U000027B0"  # Dingbats
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\u3030"
            "\ufe0f"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def remove_url_helper(self, string):
        """
        Purpose: 
            - Helper function to remove url from a string 
        
        Args: 
            - string (str): string to parse
        
        Returns: 
            - returns string without url
        """
        
        url_pattern = r"https?://\S+"
        return re.sub(url_pattern, '', string)

    def process_array_string(self, string):
        """
        Purpose: 
            - Helper function to convert a list that is formatted as a string into a list
        
        Args: 
            - string (str): string representing array to be converted into an array 
        
        Returns: 
            - values (list_like): list of all parsed values in the string 
        """
        
        values = [float(val) for val in string[1:-1].split(',')]
        return values
    
    def format_str(self, string): 
        """
        Purpose: 
            - Helper function to remove white space from a string 
            
        Args: 
            - string (str): string to be editted
        
        Returns: 
            - fstring (str): string that is removed of all white space 
        """
        
        fstring = re.sub('\s+',' ', string)
        fstring = fstring.strip()
        
        return fstring
    
    @time_function 
    def filter_dataframe(self, filter_col, filter_val): 
        """
        Purpose: 
            - Given a dataframe extract rows where the a column's (filter_col) matches a value (filter_val) 
        
        Args: 
            - filter_col (str): column name to be filtered 
            - filter-_val (any): value to be matched 
        
        Returns: 
            - None: current instance's dataframe attribute (self.df) is updated to only contain rows in which the filter_col values match the filter_val
        """
        
        self.df = self.df[~self.df[filter_col].isnull()].copy()
        self.df = self.df[self.df[filter_col] == filter_val].copy()

    @time_function
    def process_text_all(self, text_col='text'):
        """
        Purpose: 
            - Function that applies the non ascii characters and emojis from text 
        
        Args: 
            - text_col (str): column to clean
        
        Returns: 
            - text (pd.Series): column removed of special characters 
        """
        
        text = self.df[text_col].apply(lambda x: self.format_str(x))
        text = text.apply(lambda x: self.remove_non_ascii(x))
        text = text.apply(lambda x: self.remove_emojis(x))
        
        return text
        
    @time_function 
    def process_text_emojis(self, text_col='text'): 
        """
        Purpose: 
            - Function that removes emojis from text 
        
        Args: 
            - text_col (str): column to remove emojis 
        
        Returns: 
            - text (pd.Series): column removed of emojis
        """
        
        text = self.df[text_col].apply(lambda x: self.remove_emojis(x))
        text = text.apply(lambda x: self.format_str(x))
        
        return text 

    @time_function
    def extract_url(self, col_name=None):
        """
        Purpose: 
            - Function that applies the url extractor helper function 
        
        Args: 
            - col_name (str): column that you would like to extract urls from 
        
        Returns: 
            - None: adds the extracted urls as a column
        """
        
        self.extract_url_col = True
        
        if not col_name:
            urls = self.df['text'].apply(self.extract_url_helper)
        else:
            urls = self.df[col_name].apply(self.extract_url_helper)

        self.df['extracted_urls'] = urls.apply(self.parse_url_helper)
    
    @time_function
    def remove_url(self, col_name=None):
        """
        Purpose: 
            - Function that applies the url extractor helper function 
        
        Args: 
            - col_name (str): column that you would like to extract urls from 
        
        Returns: 
            - None: column with urls removed from text
        """
        
        self.no_url = True
        
        if not col_name:
            text = self.df['text'].apply(self.remove_url_helper)
        else:
            text = self.df[col_name].apply(self.remove_url_helper)

        self.df['text_no_urls'] = text
    

    @time_function
    def process_embeddings(self, embd_col=None):
        """
        Purpose: 
            - Given a column name with embeddings, extracts the embeddings to create a numpy array 
        
        Args: 
            - embd_col (str): column to get embeddings from 
        
        Returns: 
            - data (np.ndarray): numpy array of all embeddings  
        """
        
        if not embd_col:
            self.df['embeddings'] = self.df['embeddings'].apply(
                self.process_array_string)
            self.data = np.vstack(self.df['embeddings'])
        else:
            self.df[embd_col] = self.df[embd_col].apply(
                self.process_array_string)
            self.data = np.vstack(self.df[embd_col])
        
        return data

    @time_function
    def create_embeddings(self, text=None, text_col='text', just_emoji = False, embd_model_path=None):
        """
        Purpose: 
            - Create embeddings from text, and sets the instance data variable to the embeddings created 
        
        Args: 
            - text (list[str]): list of text to create embeddings from 
            - embd_model_path (str): path to embedding model directory 
        
        Returns: 
            - data (np.ndarray): numpy array of all embeddings  
        """
        
        if not embd_model_path:
            model = SentenceTransformer(self.embd_model_path, device='cuda')
        else:
            model = SentenceTransformer(file_path,  device='cuda')

        if text:
            text = list(text)
            data = model.encode(text)
        else:
            
            if just_emoji: 
                text = self.process_text_emojis(text_col).to_list()
            else: 
                text = self.process_text_all(text_col).to_list()
                
            data = model.encode(text)

        self.data = data
        self.df['embeddings'] = data.tolist()

        return data

    @time_function
    def reduce_dimensions(self, data=None, n_components=3, n_neighbors=None, min_dist=None):
        """
        Purpose: 
            - Given a list of text embeddings, reduce dimensions (default dimensions is 3)  
        
        Args: 
            - data (list/array like): numpy array of text embeddings 
            - n_components (int): number of dimensions to be reduced down to 
            - n_neighbors (int): number of neighbors to consider when reducing dimensions 
            - min_dist (int): minimum distance between plots when reducing dimensions 
        
        Returns: 
            - reduced_data (np.ndarray): numpy array of all dimensionally reduced embeddings 
        """
        
        self.rd = True
        
        if data is None: 
            if self.data is None: 
                self.create_embeddings()
            data = self.data 
        
        if n_neighbors is None: 
            n_neighbors = 15 
        
        if min_dist is None: 
            min_dist = 0.1 
        
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=8)
        
        reduced_data = reducer.fit_transform(data) 
        print(f'\t- Reduced data shape: {reduced_data.shape}')
        
        self.data = reduced_data 
        cols = [f'Dim{i}' for i in range(reduced_data.shape[-1])]
        tmp_df = pd.DataFrame(reduced_data, columns=cols)
        tmp_df.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f'\t- Temporary DataFrame Shape: {tmp_df.shape}')
        print(f'\t- Self DataFrame Shape Before Concatenation: {self.df.shape}')
        self.df = pd.concat([self.df, tmp_df], axis=1)
        print(f'\t- Self DataFrame Shape After Concatenation: {self.df.shape}')
        
        return reduced_data 
    
    
    @time_function
    def sdedup(self, data=None, n_clusters=600, threshold=0.9999): 
        """
        Purpose: 
            - Remove semantically similar text
        
        Args: 
            - data (list/array like): data to remove semantically similar items 
        
        Returns: 
            - similarity (pd.Series): series with true or false values stating whether something is semantically similar or not 
        """
        
        self.semdup = True 
        
        if self.rd: 
            cols = [f'Dim{i}' for i in range(self.data.shape[-1])]
            dims = self.data
        else: 
            dims = self.reduce_dimensions()
                    
        print(f'\t -Shape: {dims.shape}')
        clusterer = KMeans(n_clusters=n_clusters) 
        labels = clusterer.fit_predict(dims) 
        
        tmp = pd.DataFrame({
            'embeddings': dims.tolist(), 
            'klabels': labels
        })          
        
        tmp = tmp.groupby('klabels')[tmp.columns].apply(lambda x: self.calc_dist_to_centroid(x, clusterer)).reset_index(drop=True)
        similarity = tmp['cosine_sim'] > threshold 
        self.df['Drop'] = ~similarity
        
        return similarity
                                                    
                                                    
                
    @time_function
    def create_clusters(self, data=None, min_cluster_size=None, min_samples=None, epsilon=None, n_components=None, n_neighbors=None, min_dist=None):
        """
        Purpose: 
            - Create clusters given a set of points/vectors, should the number of components be specified dimension reduction will be performed and 
            then clustered 
            
        Args: 
            - data (ndarray): array of vectors to be clustered 
            - min_cluster_size (int): minimum number of points for a grouping to be considered a cluster 
            - min_samples (int): minimum number of points within a threshold for a point to be considered a core point 
            - n_components (int): number of dimensions to be reduced down to 
            - n_neighbors (int): number of neighbors to consider when reducing dimensions 
            - min_dist (int): minimum distance between plots when reducing dimensions 
            
        Returns: 
            - labels (ndarray): numpy array of labels associated with a vector
        """
        
        if data is None:
            if self.data is None: 
                self.create_embeddings() 
            
            data = self.data 
        
        print(f'\t- Data is clustered on shape: {data.shape}')
        
        if n_components is not None:
            data = self.reduce_dimensions(data=self.data, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components) 
        
        if min_cluster_size is None: 
            min_cluster_size = 15 
            
        if min_samples is None: 
            min_samples = min_cluster_size 
        
        if epsilon is None: 
            epsilon = 0.0 
                
        if min_cluster_size is None: 
            min_cluster_size = 15 
                
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=epsilon)
        print(f'\t- Labels shape {data.shape}')
        print(f'\t- DataFrame shape {self.df.shape}')
        
        labels = clusterer.fit_predict(data)
        self.df['clust_id'] = labels
        print(f'\t- Number of clusters: {self.df.clust_id.nunique()}')
        return labels

    @time_function
    def to_csv(self, output_path, cols=None):
        """
        Purpose: 
            - Generate output csv file 
        
        Args: 
            - output_path (str): path for output file 
            - cols (list): list of strings for column names to be extracted from dataframe to convert into a csv  
            
        Returns: 
            - None
        """
        
        if not cols:
            self.df['text'] = self.df['text'].apply(lambda x: self.format_str(x))
            columns = ['text', 'clust_id']
            
            if self.rd:
                dimensions = [f'Dim{i}' for i in range(self.data.shape[-1])] 
                columns += dimensions 
            
            if self.no_url: 
                columns += ['text_no_urls']
            
            if self.extract_url_col: 
                columns += ['extracted_urls'] 
                
            if self.semdup: 
                columns += ['Drop']
                
            self.df[columns].to_csv(
                output_path, index=False)
        else:
            self.df[cols].to_csv(output_path, index=False)
