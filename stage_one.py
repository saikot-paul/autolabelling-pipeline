import fire
import time
from preprocess_data import DataProcessor 

def main(
    file_path: str = None, 
    embd_model_path: str = None, 
    output_path: str = None, 
): 
    
    print('=================================================================')
    start = time.time()
    print('Processing File Now')
    dp = DataProcessor(file_path=file_path, embd_model_path=embd_model_path)
    dp.process_text_all()
    dp.create_embeddings()
    dp.reduce_dimensions(n_components=2)
    dp.binary_search_hdbscan()
    dp.to_csv(output_path) 
    end = time.time() 
    print(f'Total time elapsed {(end - start):0.2f}')

if __name__ == '__main__': 
    fire.Fire(main)
