import csv
import sys
import ast
import math
csv.field_size_limit(sys.maxsize)
embedings_file = sys.argv[1]

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

embeddings = []
with open(embedings_file, 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = '\n')
    vectors = []
    for row in data:
        row = row[0]
        row= row.split('JPG')
        img_name = row[0] + 'JPG'
        embeding = row[1].strip()
        embeding = embeding.replace(' ', ',')
        embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
        embeddings.append(embeding)


    for index, emb in enumerate(embeddings):
        if index < len(embeddings) - 1 :
            print(cosine_similarity(emb, embeddings[index + 1]))

