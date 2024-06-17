"""
Previous prompts

prompt_p1 = 
        The following body of text represents a cluster, I want you to create a descriptive title for the text using the following framework:\n
        1. Analyze the text\n
        2. Identify common themes\n 
        3. Using the information gained from the steps above create a descriptive title that generalizes the text in 20 words or less
        \n\nText to label: \n

prompt_p2 =
        \n\nTitle: 


-- FIRST ONE
system_prompt = You are a data annotator. You will be given text that is representative of a cluster, your job is to create a label of the representative text. The created a label will be judged on the following criteria:\n\n\t1. Descriptiveness of label.\n\t2. Conciseness of the label.\n\t3. Informativeness of the label.\n\t4. How well the label aligns with the content of the representative text.\n\nReturn the label for the given body of text in this format: \n\nLabel: "...".

-- SECOND ONE
system_prompt = You are a data annotator. You will be given text that is representative of a cluster, your job is to create a label of the representative text. The following label will be judged on the following criteria: \n\n\t1. Informativeness\n\t2. Concise\n\t3. Generalizes the content of the representative text\n\t4. Less than 10 words.\n\t5. How well the label aligns with the overall content of the representative text.\n\nReturn ONLY the label for the given body of text in this format: \n\nLabel: ".....".

-- THIRD ONE 
system_prompt = You are a data annotator. You will be given text that is representative of a cluster, your job is to create a label of the representative text. Create a label such that it is: descriptive, concise, aligns with the content representative text. \nReturn the label for the given body of text in this format: \n\nLabel: ".....". 

COVE QUESTION 

-- TRIAL 1 
questions = [
                'Is this label informative of the representational text originally provided?', 
                'Does the label accurately identify the common themes and topics spoken in the representational text?', 
                'Does the label accurately identify the narratives in the representational text?', 
                'What is done well and what could be done better?', 
                'Gathering the answers and insight gained from the previous questions recreate a SINGLE label  that is less than or equal to 15 words for the representational text.\nReturn in the format\n Label: "..."'
            ] 
            
-- TRIAL 2 
questions = [
                "What are the main themes, messages and topics discussed in the representational text? Keep answers concise.", 
                "What are the main narratives in the representational text? Keep answers concise.", 
                "What are the top 10 keywords and/or entities spoken about in the representational text? Keep answers concise.", 
                'Using the information gained from the previous questions create a label. The label should:\n\t1. Accurately reflect themes, messages and topics described in the representational text.\n\t2. Accurately portray narratives in the representational text.\n\t3. Use keywords and entities from the representational text.\n\t4. Be non-dramatic.\n\t5. Be less than or equal to 15 words\nReturn a SINGLE label in the format\nLabel: "..."'
            ] 

"""

prompt_p1 = """The following body of text represents a grouping, each piece of text is appended with: ***. They have been grouped together due to sharing common themes, meaning and sentiment. Given this cluster of text, create a description using the following framework:\n\n\t1. Analyze the text.\n\t2. Identify shared themes: Look for reccuring ideas, concepts and key messages.\n\t3. Identify shared topics/subjects.\n\t4. Identify the sentiment: Define the overall mood/tone of the text.\n\t5. Identify the narrative \n\t6. Generate a concise, descriptive and informative label that captures the essence, messages, theme and of the text using the information gained by following the framework.\n\nRepresentational Text/Original Text:
"""

prompt_p2 = """Return only the label in the following format\nLabel: "..."."""


sys_p1 = """You are a data annotator. You will be given text that is representative of a cluster. Your job is to create a label for the representative text. The created label will be judged on the following criteria:\n\n\t1. Descriptiveness of label.\n\t2. Conciseness of the label.\n\t3. Informativeness of the label.\n\t4. How well the label aligns with the content of the representative text.\n\nReturn the label for the given body of text in this format: \n\nLabel: "...".
"""

sys_p2 = """You are a data annotator. You will be given text that is representative of a cluster. Your job is to create a label for the representative text. The following label will be judged on the following criteria: \n\n\t1. Informativeness\n\t2. Conciseness\n\t3. Generalizes the content of the representative text\n\t4. Less than 10 words.\n\t5. How well the label aligns with the overall content of the representative text.\n\nReturn ONLY the label for the given body of text in this format: \n\nLabel: ".....".
"""

sys_p3 = """You are a data annotator. You will be given text that is representative of a cluster. Your job is to create a label for the representative text. Create a label such that it is: descriptive, concise, and aligns with the content of the representative text. \nReturn the label for the given body of text in this format: \n\nLabel: ".....".
"""

questions_c1 = [
                'Is this label informative of the representational text originally provided?', 
                'Does the label accurately identify the common themes and topics spoken in the representational text?', 
                'Does the label accurately identify the narratives in the representational text?', 
                'What is done well and what could be done better?', 
                'Gathering the answers and insight gained from the previous questions recreate a SINGLE label  that is less than or equal to 15 words for the representational text.\nReturn in the format\n Label: "..."'
            ] 

questions_c2 = [
                "What are the main themes, messages and topics discussed in the representational text? Keep answers concise.", 
                "What are the main narratives in the representational text? Keep answers concise.", 
                "What are the top 10 keywords and/or entities spoken about in the representational text? Keep answers concise.", 
                'Using the information gained from the previous questions create a label. The label should:\n\t1. Accurately reflect themes, messages and topics described in the representational text.\n\t2. Accurately portray narratives in the representational text.\n\t3. Use keywords and entities from the representational text.\n\t4. Be non-dramatic.\n\t5. Be less than or equal to 15 words\nReturn a SINGLE label in the format\nLabel: "..."'
            ] 