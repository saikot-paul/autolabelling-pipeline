import fire
import pandas as pd
import numpy as np

def get_sample_whole(wdf, ldf1, ldf2, cluster, lbl1='CoVe_Q1', lbl2='CoVe_Q2', text='Text'):
    
    wdf_tmp = wdf[wdf['clust_id'] == cluster] 
    sample_text = " **** ".join(wdf_tmp['text'].sample(frac=0.5).to_list())
    lbl_1 = ldf1[ldf1['Cluster'] == cluster][lbl1].values[0]
    lbl_2 = ldf2[ldf2['Cluster'] == cluster][lbl2].values[0]
    rep_text = ldf2[ldf2['Cluster'] == cluster][text].values[0]
    
    return (sample_text, lbl_1, lbl_2, rep_text)


def main(wdf_path : str, df1_path: str, df2_path: str, agg_file_name: str): 
    
    wdf = pd.read_csv(wdf_path)
    ldf = pd.read_csv(df1_path)
    ldf_q2 = pd.read_csv(df2_path)

    cids = ldf.Cluster.unique()[1:]

    cluster = []
    rndm_text = []
    rep_text = []
    cove_one = [] 
    cove_two = [] 

    for c in cids: 
        rd_text, lb1, lb2, r_text = get_sample_whole(wdf, ldf, ldf_q2, c)
        cluster.append(c) 
        rndm_text.append(rd_text) 
        cove_one.append(lb1) 
        cove_two.append(lb2) 
        rep_text.append(r_text)

    df = pd.DataFrame({ 
        'Cluster': cluster,
        'Rep Text': rep_text, 
        'Random Text': rndm_text, 
        'CoVe_Q1_Label': cove_one, 
        'CoVe_Q2_Label': cove_two, 
    })

    df.to_csv(agg_file_name, index=False)


if __name__ == '__main__': 
    fire.Fire(main)