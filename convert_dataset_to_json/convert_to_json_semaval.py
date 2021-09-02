import json
from lxml import etree

output_file_name = "semaval.data.json"
input_file_name = "SemEval2017-task3-English-test-labeled.xml"

queries = {}
class document:
    def __init__(self, candidate_question, label):
        self.candidate_question = candidate_question
        self.label = int(label)
        
root = etree.parse(input_file_name)
original_questions = root.findall("OrgQuestion")
for original_question in original_questions:
    question_body = original_question.find("OrgQBody")
    related_question = original_question.find("Thread/RelQuestion")
    related_question_body = original_question.find("Thread/RelQuestion/RelQBody")
    if question_body.text not in queries.keys():
        queries[question_body.text] = []
    
    related_q_text = related_question_body.text
    related_q_relevance = None
    if related_question.attrib['RELQ_RELEVANCE2ORGQ'] == "PerfectMatch":
        related_q_relevance = 1
    elif related_question.attrib['RELQ_RELEVANCE2ORGQ'] == "Relevant":
        related_q_relevance = 1
    elif related_question.attrib['RELQ_RELEVANCE2ORGQ'] == "Irrelevant":
        related_q_relevance = 0
    queries[question_body.text].append(document(related_q_text, related_q_relevance))


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
