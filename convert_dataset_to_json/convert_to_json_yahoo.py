import csv
import json
import random

output_file_name = "yahoo.data.json"
input_file_name = "yahoo.data"

queries = {}
class document:
    def __init__(self, candidate_question, label, unique_key):
        self.candidate_question = candidate_question
        self.label = int(label)
        self.unique_key = unique_key
        
with open(input_file_name, encoding="utf-8") as f_in:
    for line in csv.reader(f_in, dialect="excel-tab"):
        if line[0] not in queries.keys():
            queries[line[0]] = []
        candidate_question = document(line[1], line[2], line[3])
        queries[line[0]].append(candidate_question)

output = {}
output["rankingProblems"] = []
maximum_document = set()
maximum_query = ""
for query in queries.keys():
    curr_query = {}
    curr_query["queryText"] = query
    curr_query["documents"] = []
    maximum_document.add(len(queries[query]))
    for document in queries[query]:
        curr_doc = {}
        curr_doc["relevance"] = document.label
        curr_doc["docText"] = document.candidate_question
        curr_query["documents"].append(curr_doc)
    output["rankingProblems"].append(curr_query)


with open(output_file_name, "w") as write_file:
    json.dump(output, write_file, indent = 2)

print("unique query num:", len(queries.keys()))
print("maximum_document:", maximum_document)
