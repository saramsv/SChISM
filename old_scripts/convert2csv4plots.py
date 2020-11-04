import os
import csv
import json
import glob


print("donor_id,", "alpha,", "beta,", "stake,", "foot,", "head,","fullbody,", "plastic,", "torso,","arm,", "legs,", "backside,", "hand")
for file_ in glob.glob("evaled*"):
    with open(file_) as fp:
        data = csv.reader(fp)
        i = 0
        final_row = []
        final_row.append(file_)
        final_row.append(file_.split("_")[2])
        final_row.append(file_.split("_")[3])
        for row in data:
            acc = {"donor_id": 0, "stake": 0, "foot": 0, "head": 0,"fullbody": 0, "plastic":0 , "torso": 0, "arm": 0, "legs": 0, "backside": 0, "hand": 0}
            if i == 1:
                for r in row:
                    tag= r.strip().strip('{').strip('}').split(":")[0].strip("'")
                    val = r.strip().strip('{').strip('}').split(":")[1].strip()
                    acc[tag] = val
                final_row.append((acc["stake"], acc["foot"], acc["head"],acc["fullbody"], acc["plastic"], acc["torso"],acc["arm"], acc["legs"], acc["backside"], acc["hand"]))
            i += 1
            
        print(final_row)
