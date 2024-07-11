import functools
import logging
import re
import time
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN, KMeans
from functools import wraps
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer
from umap import UMAP

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
        self.read_file()
        
        
        file_name = file_path.split('.')[0]
        log_file = f'{file_name}.log'
        
        logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

        self.logger = logging.getLogger(__name__)

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
            self.df = pd.read_csv(self.file_path).dropna(
                subset=col_to_drop_from).reset_index(drop=True)
        else:
            self.df = pd.read_csv(self.file_path)

    def remove_url_helper(self, string):
        """
        Purpose:
            - Helper function to remove url from a string

        Args:
            - string (str): string to parse

        Returns:
            - returns string without url
        """

        try:
            url_pattern = r"https?://\S+"
            return re.sub(url_pattern, '', string)
        except:
            return ''

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

    def get_length_helper(self, string):
        """
        Purpose:
            - Helper function that gets the length of a string stripped of its whitespace

        Args:
            - string (str): string to have length extracted from

        Returns:
            - length (int): length of string
        """

        s = str(string).strip().split()
        length = len(s)

        return length

    def reddit_remove_helper(self, string):
        """
        Purpose: 
            - Helper function to check if string is a removed/deleted post on reddit 

        Args: 
            - string (str)

        Returns: 
            - (bool): variable expressing whether it is removed/deleted
        """

        return string in ['[removed]', '[deleted]']

    def get_language_helper(self, string):
        """
        Purpose: 
            - Given a string detect the language

        Args: 
            - string (str): string to be checked for language 

        Returns: 
            - lang (str): language that the string belongs to 
        """

        try:
            lang = detect(string)
        except:
            lang = 'error'

        return lang

    def check_language_helper(self, string):
        """
        Purpose: 
            - Given a string determine whether the language is compatible with Meta-Llama-3 

        Args: 
            - string (str): string to examined for compatibality 

        Returns: 
            - lang_check (bool): variable representing compatibality 
        """

        lang_list = ["en", "es", "fr", "de", "it",
                     "pt", "nl", "ru", "zh", "ja", "ko"]
        lang_check = string in lang_list

        return lang_check

    def remove_emoji_helper(self, text):
        """
        Purpose: 
            - Given a string remove the emojis from it 

        Args: 
            - text (str): text to have emojis removed 

        Returns: 
            - (str): string without any emojis 
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Additional symbols and pictographs
            "\U0001f926-\U0001f937"  # Gestures and people related emojis
            "\U00010000-\U0010ffff"  # Supplementary characters in unicode
            "\u2640-\u2642"          # Male and female symbols
            "\u2600-\u2B55"          # Weather and geometry
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\u3030"
            "\ufe0f"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def format_str_helper(self, string):
        """
        Purpose: 
            - Given a string remove any whitespace/line breaks

        Args: 
            - string (str): text to be editted

        Returns: 
            - fstring (str): editted string 
        """

        fstring = re.sub('\s+', ' ', string)
        fstring = fstring.strip()

        return fstring

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

    @time_function
    def extract_url(self, col_name='text'):
        """
        Purpose: 
            - Function that extracts urls from a column and then parses to get the words in the url 

        Args: 
            - col_name (str): column that you would like to extract urls from 

        Returns: 
            - None: adds the extracted urls as a column
        """

        self.extract_url_col = True

        urls = self.df[col_name].apply(lambda x: self.extract_url_helper(x))
        self.df['extracted_urls'] = urls.apply(self.parse_url_helper)

    @time_function
    def remove_url(self, col_name='text'):
        """
        Purpose: 
            - Function that removes url given the name of a column  

        Args: 
            - col_name (str): column that you would like to remove urls from 

        Returns: 
            - None: column with urls removed from text
        """

        self.no_url = True

        text = self.df[col_name].apply(lambda x: self.remove_url_helper(x))
        self.df['text_no_urls'] = text

        return text

    @time_function
    def get_length(self, col_name='text'):
        """
        Purpose: 
            - Function that gets the length of each string given a column name

        Args: 
            - col_name (str): column that you would like to get urls from

        Returns: 
            - length (pd.Series): column containing the length of text for each row in col_name
        """

        self.df['length'] = self.df[col_name].apply(
            lambda x: self.get_length_helper(x))

        return self.df['length']

    @time_function
    def reddit_remove_check(self, col_name='text'):
        """
        Purpose: 
            - Function that checks whether items in a column are removed/deleted posts 

        Args: 
            - col_name (str): column that you would like to get urls from

        Returns: 
            - reddit_removed (pd.Series): column containing boolean variable on whether something is a removed/deleted post
        """

        reddit_removed = self.df[col_name].apply(
            lambda x: self.reddit_remove_helper(x))
        self.df['reddit_removed'] = reddit_removed

        return reddit_removed

    @time_function
    def process_text_all(self, text_col='text'):
        """
        Purpose: 
            - Function that applies the non ascii characters and emojis from text 
            - Drops all rows that are empty strings after filtering process and non-compatible languages 

        Args: 
            - text_col (str): column to clean

        Returns: 
            - tmp_df (pd.DataFrame): cleaned dataframe 
        """

        def filters(*condition):
            return functools.reduce(np.logical_and, condition)

        start = time.time()
        print('\t- Removing URLS')
        text = self.df[text_col].apply(lambda x: self.remove_url_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Removing Emojis')
        text = text.apply(lambda x: self.remove_emoji_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Formatting strings')
        text = text.apply(lambda x: self.format_str_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Getting length')
        length = text.apply(lambda x: self.get_length_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Checking for reddit removed')
        reddit_check = text.apply(lambda x: self.reddit_remove_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Getting languages')
        langs = text.apply(lambda x: self.get_language_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        start = time.time()
        print('\t- Checking languages')
        lang_check = langs.apply(lambda x: self.check_language_helper(x))
        end = time.time()
        print(f'\t\t- Time taken: {end-start:.2f}s')

        tmp_df = pd.DataFrame({
            'text': text,
            'lang': langs,
            'length': length,
            'lang_check': lang_check,
            'reddit_check': reddit_check
        })

        c1 = length > 2
        c2 = ~reddit_check
        c3 = lang_check

        tmp_df = tmp_df[filters(c1, c2, c3)]
        tmp_df = tmp_df.reset_index()

        self.df = tmp_df.copy()

        return tmp_df

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

        return self.data

    @time_function
    def create_embeddings(self, text=None, text_col='text', embd_model_path=None):
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

        if text is not None:
            data = model.encode(text)
        else:
            text = self.df[text_col].to_list()
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

        reducer = UMAP(n_components=n_components,
                       n_neighbors=n_neighbors, min_dist=min_dist, random_state=8)

        reduced_data = reducer.fit_transform(data)
        print(f'\t- Reduced data shape: {reduced_data.shape}')

        self.data = reduced_data
        cols = [f'Dim{i}' for i in range(reduced_data.shape[-1])]
        tmp_df = pd.DataFrame(reduced_data, columns=cols)
        tmp_df.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f'\t- Temporary DataFrame Shape: {tmp_df.shape}')
        print(
            f'\t- Self DataFrame Shape Before Concatenation: {self.df.shape}')
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

        tmp = tmp.groupby('klabels')[tmp.columns].apply(
            lambda x: self.calc_dist_to_centroid(x, clusterer)).reset_index(drop=True)
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

        if not self.rd and n_components is not None:
            data = self.reduce_dimensions(
                n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)

        if min_cluster_size is None:
            min_cluster_size = 15

        if min_samples is None:
            min_samples = min_cluster_size

        if epsilon is None:
            epsilon = 0.0

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples, cluster_selection_epsilon=epsilon)

        print(f'\t- Shape of data to cluster: {data.shape}')
        print(f'\t- Clusterer Params:')
        print(f'\t\t- Min Cluster Size: {min_cluster_size}')
        print(f'\t\t- Min Sample Size: {min_samples}')

        labels = clusterer.fit_predict(data)
        lbls, counts = np.unique(labels, return_counts=True)

        num_outliers = counts[lbls == -1].sum() if -1 in lbls else 0
        num_clusters = counts[lbls != -1].shape[0]

        size = data.shape[0]
        percentage_outliers = (num_outliers / size)*100

        print(f'\t- Clustering Statistics: ')
        print(f'\t\t- Number of Clusters: {num_clusters}')
        print(f'\t\t- Percentage Outliers: {percentage_outliers}')
        print(f'\t\t- Number of Outliers: {num_outliers}')

        return num_clusters, num_outliers, percentage_outliers, labels

    @time_function
    def binary_search_hdbscan(self):

        def get_bounds(size):
            if size <= 100:
                return 5, 10
            elif size <= 1000:
                return 10, 50
            elif size <= 10000:
                return 20, 80
            else:
                return 50, 120

        size = len(self.df)
        lower, upper = get_bounds(size)
        l, r = 5, 1023

        num_clusters = []
        num_outliers = []
        percentage_outliers = []
        min_csize = []
        min_s = []
        labels_arr = []

        # Searching parameter space
        while l <= r:

            min_cluster_size = l + (r - l)//2
            min_samples = min_cluster_size // 2

            n_clust, n_outliers, p_outliers, labels = self.create_clusters(
                data=self.data, min_cluster_size=min_cluster_size, min_samples=min_cluster_size)

            num_clusters.append(n_clust)
            num_outliers.append(n_outliers)
            percentage_outliers.append(p_outliers)
            min_csize.append(min_cluster_size)
            min_s.append(min_cluster_size)
            labels_arr.append(labels)

            n_clust, n_outliers, p_outliers, labels = self.create_clusters(
                data=self.data, min_cluster_size=min_cluster_size, min_samples=min_samples)

            num_clusters.append(n_clust)
            num_outliers.append(n_outliers)
            percentage_outliers.append(p_outliers)
            min_csize.append(min_cluster_size)
            min_s.append(min_samples)
            labels_arr.append(labels)

            # More clusters than needed, need less clusters, increase minimum cluster size to decrease number of clusters
            if n_clust > upper:
                l = min_cluster_size + 1
            # Too little clusters than needed, decrease minimum cluster size to increase number of clusters
            elif n_clust < lower:
                r = min_cluster_size - 1
            else:
                break

        # Finding optimal parameters
        scores = pd.DataFrame({
            'Min Cluster Size': min_csize,
            'Min Samples': min_s,
            'Number of Clusters': num_clusters,
            'Number of Outliers': num_outliers,
            'Percentage Outliers': percentage_outliers,
            'Labels': labels_arr
        })

        condition = (scores['Percentage Outliers'] <= 25) & (
            scores['Percentage Outliers'] > 0)
        filtered = scores[condition]

        if (filtered.shape[0] == 0):
            threshold = scores[scores['Number of Clusters'] > lower]
            params = threshold.loc[threshold['Percentage Outliers'].idxmin()]
            labels = params['Labels']
        else:
            params = filtered.loc[filtered['Number of Clusters'].idxmax()]
            labels = params['Labels']

        self.df['clust_id'] = labels
        
        print(f"\t- Params Chosen: ")
        print(f"\t\t- Min Cluster Size: {params['Min Cluster Size']}")
        print(f"\t\t- Min Samples: {params['Min Samples']}")
        print(f"\t\t- Number of Clusters: {params['Number of Clusters']}")
        print(f"\t\t- Percentage Outliers: {params['Percentage Outliers']}")
        print(f"\t\t- DataFrame value counts: {self.df.clust_id.value_counts()}")


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

        print(output_path)
        if not cols:
            self.df['text'] = self.df['text'].apply(
                lambda x: self.format_str_helper(x))
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
