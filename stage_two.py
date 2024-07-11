import fire 
from generate_labels import LabelGenerator

def main( 
        ckpt_dir: str = None,
        tokenizer_path: str = None, 
        max_seq_len: int = None, 
        max_batch_size: int = None, 
        file_path: str = None, 
        output_path: str = None, 
        temperature: float = None, 
        text_col: str = None,
        sys_id: int = None, 
        user_id: int = None,
        cove_id: int = None, 
        label_col: str = None
): 
    
    lb = LabelGenerator(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size, 
        temperature=0.0, 
        output_path=output_path,
        sys_id = sys_id, 
        user_id = user_id, 
        cove_id = cove_id, 
        label_col = label_col
    )
    
    print(f'TEXT COL: {text_col}')
    lb.read_file(file_path, drop_na_col='text')
    lb.generate_labels_pipeline(text_col=text_col, random=False)
    lb.generate_output()

if __name__ == '__main__': 
    fire.Fire(main)