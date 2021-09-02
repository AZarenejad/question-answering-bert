
import json


def averge_precesion(query):
    corroctPredictions = 0
    runningSum = 0
    total_relevance_documents = 0
    k = 10
    for i in range(len(query["documents"])):
        document = query["documents"][i]
        if i + 1 > k:
            break
        if document["relevance"] == 0:
            break
        total_relevance_documents += 1
        corroctPredictions +=1
        runningSum = runningSum + float(corroctPredictions/(i + 1))
    if total_relevance_documents != 0:
        ap = float(runningSum) / total_relevance_documents
        return ap
    return 0
    

input_scores_file = "eval.scoresOut.json"

# Opening JSON file
f = open(input_scores_file, 'r')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
queries = data["rankingProblemsOutput"]
print("total question:", len(queries))
map = 0
for query in queries:
    map += averge_precesion(query)
map = float(map) / len(queries)
print("map:", map)


