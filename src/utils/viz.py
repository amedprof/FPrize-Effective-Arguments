import spacy
from spacy import displacy
from pylab import cm, matplotlib
import os

colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000'
         }

def visualize(idx,train):
    ents = []
    for i, row in train[train['essay_id'] == idx].iterrows():
        ents.append({
                        'start': int(row['discourse_start']), 
                         'end': int(row['discourse_end']), 
                         'label': row['discourse_type']
                    })

    data = train[train['essay_id'] == idx].essay_text.values[0]

    doc2 = {
        "text": data,
        "ents": ents,
        "title": idx
    }

    options = {"ents": train.discourse_type.unique().tolist(), "colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)