import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")  

file_path = 'C:/Users/adyab/OneDrive/Desktop/NLP/NLP project/cleaned_train.csv'
data = pd.read_csv(file_path)


updated_data = {
    'word': [],
    'label': []
}


for words, labels in zip(data['words_list'], data['labels']):
    word_list = words.strip("[]").replace("'", "").split()
    label_list = labels.strip("[]").replace("'", "").split()
    doc = nlp(" ".join(word_list))  
    gpe_entities = {ent.text for ent in doc.ents if ent.label_ == "GPE"}
    
    
    for word, label in zip(word_list, label_list):
        clean_word = word.strip()
        if clean_word.startswith('@'):
            updated_data['word'].append(clean_word)
            updated_data['label'].append('handle')
        elif clean_word in gpe_entities:
            updated_data['word'].append(clean_word)
            updated_data['label'].append('name')
        else:
            updated_data['word'].append(clean_word)
            updated_data['label'].append(label)


updated_df = pd.DataFrame(updated_data)


output_file = 'C:/Users/adyab/OneDrive/Desktop/NLP/NLP project/predicted.csv'
updated_df.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}")
