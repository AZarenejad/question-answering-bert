import pandas as pd
import json, ssl
from urllib.request import Request, urlopen
import requests
from bs4 import BeautifulSoup
from lxml import etree
from requests_html import HTMLSession
from w3lib.url import safe_url_string
import argparse
import time
import unicodedata

INPUT_FILE_NAME = './quora_duplicate_questions.tsv'
# OUTPUT_FILE_NAME = 'output.xml'
QUORA_SITE = "https://www.quora.com/"

session = HTMLSession()


def add_status(question, is_found):
    question_status = etree.SubElement(question, "status")
    if is_found:
        question_status.text = "page found"
    else:
        question_status.text = "page not found"

def add_question_part(pair, text, part_number):
    question = etree.SubElement(pair, "Question" + str(part_number))
    question_text = etree.SubElement(question, "Question_text")
    question_text.text = text
    return question

def add_label(pair, pair_id):
    label = etree.SubElement(pair, "label")
    label.text = pair_id

def add_user_of_question(soup):
    find_application_json = soup.find('script', type='application/ld+json')
    if find_application_json is None:
        return None
    main = json.loads(find_application_json.string)['mainEntity']
    try:
        if "author" in main:
            author = main["author"]
            return author['url']
        else:
            return None
    except Exception:
        return None

def add_topic_part(question, question_url):
    r = session.get(question_url + "/log")
    r.html.render(timeout = 100)
    soup = BeautifulSoup(r.html.html, 'html.parser')
    topic = soup.find_all("div", {'class': 'q-flex qu-flexWrap--wrap qu-alignItems--center'})
    topic_num = etree.SubElement(question, "topic_num")
    if len(topic) == 1:
        spans = topic[0].find_all("span", {'class': 'q-text'})
        topic_num.text = str(len(spans))
        for span in spans:
            t = etree.SubElement(question, "topic")
            t.text = span.text

class AnswerDetail:
    def __init__(self):
        self.author_name = ""
        self.upvote_count = 0
        self.text = ""

def answers_section(question, response):
    soup = BeautifulSoup(response.html.html, 'html.parser')
    finds = soup.find_all('div', {'class':'q-box qu-pb--medium qu-borderBottom'})
    answer_detail_list = []
    for f in finds:
        ans = f.find_all('div' , {'class': 'q-relative spacing_log_answer_content puppeteer_test_answer_content'})
        upvote = f.find_all('div', {'class': 'q-relative qu-mr--n_small qu-pr--small qu-overflowY--hidden qu-ml--tiny qu-minHeight--20 qu-color--gray qu-minWidth--24'})
        if len(ans) == 1:
            answer_detail = AnswerDetail()
            answer_detail.text = ans[0].text
            answer_detail.upvote_count = 0
            if len(upvote) == 1:
                count = 0
                if upvote[0].text[-1] == 'K':
                    count = float(upvote[0].text[:-1]) * 1000
                elif upvote[0].text[-1] == 'M':
                    count = float(upvote[0].text[:-1]) * 1000000
                else:
                    count = int(upvote[0].text)

                answer_detail.upvote_count = count
            answer_detail_list.append(answer_detail)
            
    answer_detail_list.sort(key=lambda x: x.upvote_count, reverse=True)

    answer_num = etree.SubElement(question, "answer_num")
    answer_num.text = str(len(answer_detail_list))
    if len(answer_detail_list) == 0:
        return
    best_answer = etree.SubElement(question, "best_answer")
    best_ans_str = "".join(ch for ch in str(answer_detail_list[0].text) if unicodedata.category(ch)[0]!="C")
    best_ans_str = best_ans_str.replace("\n", " ")
    best_ans_str = best_ans_str.replace("\t", " ")
    best_answer.text = best_ans_str
    best_answer_vote = etree.SubElement(question, "vote")
    best_answer_vote.text = str(answer_detail_list[0].upvote_count)

def user_question_part(question, user_link):
    r = session.get(user_link + "/questions")
    r.html.render(timeout = 100, scrolldown = 1000)
    soup = BeautifulSoup(r.html.html, 'html.parser')
    spans = soup.find_all('span', {'class' : 'q-box qu-userSelect--text'})
    user_all_questions_num = etree.SubElement(question, "user_all_questions_num")
    user_all_questions_num.text = str(len(spans))
    for span in spans:
        user_question = etree.SubElement(question, "user_question")
        user_question.text = span.get_text()

def user_answer_part(question, user_link):
    r = session.get(user_link + "/answers")
    r.html.render(timeout = 100, scrolldown = 1000)
    soup = BeautifulSoup(r.html.html, 'html.parser')
    answer_boxes = soup.find_all('div', {'class' : 'q-box qu-pt--medium qu-pb--medium qu-hover--bg--darken'})
    user_all_answers_num = etree.SubElement(question, "user_all_answers_num")
    user_all_answers_num.text = str(len(answer_boxes))
    for tag in answer_boxes:
        tdTags = tag.find_all("div", {"class": "q-relative spacing_log_answer_content puppeteer_test_answer_content"})
        for tag in tdTags:
            user_answer = etree.SubElement(question, "user_answer")
            user_answer_str = str(tag.get_text())
            user_answer_str = user_answer_str.replace("\n", " ")
            user_answer_str = user_answer_str.replace("\t", " ")
            user_answer.text = user_answer_str

def get_question_html(question_text):
    for char in "?,.()":
        question_text = question_text.replace(char, "")
    question_url = QUORA_SITE + question_text.replace(" ", "-")
    r = session.get(question_url)
    if r.status_code == 200:
        return question_url
    else:
        r = session.get(safe_url_string("https://www.quora.com/search?q=" + question_text))
        r.html.render(timeout = 20000, scrolldown = 5000)
        soup = BeautifulSoup(r.html.html, 'html.parser')
        search_result = soup.find_all('a', {'class' : 'q-box qu-display--block qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 roKEj'})
        if len(search_result) >= 1:
            return search_result[0]['href']
        else:
            return None

def get_url_of_user_from_question(url_question):
    r = session.get(url_question + "/log")
    r.html.render(timeout = 100, scrolldown = 100, sleep = 0.15)
    soup = BeautifulSoup(r.html.html, 'html.parser')
    logs = soup.find_all('div', {'class': 'q-box qu-mb--small'})
    if len(logs) == 0:
        return None
    a = logs[len(logs)-1].find_all('a')
    if len(a) != 2:
        return None
    try:
        return a[1]['href']
    except Exception:
        return None

def prepare_question_info(pair, question_text, question_id):
    # Question
    question = add_question_part(pair, question_text, question_id)

    # request to url
    url = get_question_html(question_text)
    if url is None:
        # print(question_text + " ===> FAIL")
        add_status(question, False)
        return
    try:
        url = safe_url_string(url)
        req = Request(url)
        response = urlopen(req)
        # print(url + " ===> SUCCESS")
    except Exception as e:
        # print(str(e) + " " + url + " ===> FAIL")
        add_status(question, False)
        return

    #status
    add_status(question, response.code == 200)
    if response.code != 200:
        return

    soup = BeautifulSoup(response.read(), 'html.parser')

    # Topic
    # number of about for this question and name of each topic.
    s = time.time()
    add_topic_part(question, url)
    # print("add_topic_part", time.time() - s)

    # Answers
    # Total answers to this question, best answer and upvotecount.
    s = time.time()
    r = session.get(url)
    r.html.render(timeout = 20000, scrolldown = 5000)
    answers_section(question, r)
    # print("best answer", time.time() - s)

    # user
    # Author name of question, return user url.
    s = time.time()
    user_url = add_user_of_question(soup)

    user = etree.SubElement(question, "user")
    if user_url is None:
        user_url = get_url_of_user_from_question(url)
        if user_url is None:
            # print("user not found")
            return
    # print(user_url + " ===> found")
    user.text = user_url.partition("/profile/")[2]
    # print("user_url", time.time() - s)

    # user answers
    s = time.time()
    user_answer_part(question, user_url)
    # print("user_answer_part", time.time() - s)

    # user questions
    s = time.time()
    user_question_part(question, user_url)
    # print("user_question_part", time.time() - s)


parser = argparse.ArgumentParser()
parser.add_argument("from_index")
parser.add_argument("to_index")
args = parser.parse_args()
OUTPUT_FILE_NAME = "output_" + str(args.from_index) + "_" + str(args.to_index) + ".xml"
ERROR_FILE = "error_index_" + str(args.from_index) + "_" + str(args.to_index) + ".txt"

# for ignoring SSL certificate errors in
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


data = pd.read_csv(INPUT_FILE_NAME, sep='\t')
for index, row in data.iterrows():
    try:
        #  Reading from file
        pair_id =  str(row['id'])
        qid1 = str(row['qid1'])
        qid2 = str(row['qid2'])
        first_question = str(row['question1'])
        second_question = str(row['question2'])
        is_duplicate =  row['is_duplicate']
        if index < int(args.from_index):
            continue
        # print("index: ", index)

        # Generating xml
        # Add pair
        pair = etree.Element("pair")
        pair.set("id", pair_id)
        # Add label
        add_label(pair, str(is_duplicate))

        # Question 1
        s = time.time()
        prepare_question_info(pair, first_question, 1)
        # print("prepare_question1", time.time() - s)

        s = time.time()
        prepare_question_info(pair, second_question, 2)
        # print("prepare_question2", time.time() - s)

        mypair = etree.tostring(pair, pretty_print=True)
        with open(OUTPUT_FILE_NAME, "ab") as output_file:
            output_file.write(mypair)
    except Exception as e:
        raise(e)
        with open(ERROR_FILE, "ab") as error_file:
            error_file.write((str(index) + "\n").encode())

    if index >= int(args.to_index):
        break
        
