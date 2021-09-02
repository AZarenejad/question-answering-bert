from lxml import etree
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("from_index")
parser.add_argument("to_index")
args = parser.parse_args()

page_not_found_file = open("page_not_found_" + args.from_index + "_" + args.to_index +  ".txt", "w")
index_not_found_file = open("index_not_found_" + args.from_index + "_" + args.to_index +  ".txt", "w")
user_not_found_file = open("user_not_found_" + args.from_index + "_" + args.to_index +  ".txt", "w")
input_file_name = "output_" +  args.from_index + "_" + args.to_index

def read_question(question, pair_id, question_id):
    question1_text = question.find("Question_text").text
    # STATUS
    status_text = question.find("status").text
    if status_text == "page not found":
        page_not_found_file.write(str(pair_id) + ":" + str(question_id) + "\n")
        return
    # TOPIC
    topic_num = 0
    if not question.find("topic_num").text is None:
        topic_num = int(question.find("topic_num").text)
    topics = []
    if topic_num > 0:
        for topic in question.findall("topic"):
            topics.append(topic.text)
    # ANSWER
    answer_num = int(question.find("answer_num").text)
    if answer_num > 0:
        best_answer = question.find("best_answer").text
        vote = float(question.find("vote").text)
    # USER
    user_name = question.find("user").text
    if user_name is None:
        user_not_found_file.write(str(pair_id) + ":" + str(question_id) + "\n")
        return
    user_answers = []
    user_questions = []
    user_answer_num = int(question.find("user_all_answers_num").text)
    user_question_num = int(question.find("user_all_questions_num").text)
    if user_answer_num > 0:
        for ans in question.findall("user_answer"):
            user_answers.append(ans.text)
    if user_question_num > 0:
        for q in question.findall("user_question"):
            user_questions.append(q.text)


def parseXML():
    with open(input_file_name + ".xml", 'r') as original: data = original.read()
    with open(input_file_name + "_new.xml", 'w') as modified: modified.write("<document>" + "\n" + data + "\n" + "</document>")   
  
    root = etree.parse(input_file_name + "_new.xml")

    pairs =  root.findall("pair")

    list_pair_id = []

    for pair in pairs:
        pair_id = int(pair.attrib['id'])
        label = int(pair.find("label").text)
        question1 = pair.find("Question1")
        read_question(question1, pair_id, 1)
        question2 = pair.find("Question2")
        read_question(question2, pair_id, 2)
        list_pair_id.append(pair_id)

    os.remove(input_file_name + "_new.xml")
    
    base_index = [i for i in range(int(args.from_index), int(args.to_index) + 1)]

    diff_list = list(set(base_index) - set(list_pair_id))
    diff_list.sort()

    for index in diff_list:
        index_not_found_file.write(str(index) + "\n")

parseXML()