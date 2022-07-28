import time
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import PyPDF2
import pickle
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup


def fetchLinkedInAbout(urloruname):
    profile_url = ""
    if "linkedin.com" in urloruname:
        profile_url = urloruname
    else:
        profile_url = "https://www.linkedin.com/in/" + urloruname

    # Login To LinkedIn with bot credentials
    driver = webdriver.Chrome(
        "C:\\Users\\ANIKA\\Downloads\\chromedriver_win32\\chromedriver.exe")
    driver.get("https://linkedin.com/uas/login")
    time.sleep(1)
    username = driver.find_element_by_id("username")
    username.send_keys("thesecondbit2@gmail.com")
    pword = driver.find_element_by_id("password")
    pword.send_keys("9Q1s4u$SD#Vy")
    driver.find_element_by_xpath("//button[@type='submit']").click()

    # Fetch and load the entire page
    driver.get(profile_url)
    start = time.time()
    initialScroll = 0
    finalScroll = 1000

    while True:
        driver.execute_script(
            f"window.scrollTo({initialScroll},{finalScroll})")

        initialScroll = finalScroll
        finalScroll += 1000

        time.sleep(1)

        end = time.time()

        if round(end - start) > 5:
            break

    src = driver.page_source
    soup = BeautifulSoup(src, 'lxml')
    driver.quit()

    sections = soup.find_all('section')
    about_section = -1
    for section in sections:
        if len(section.find_all('div', {'id': "about"})) > 0:
            about_section = section
    about = about_section.find_all('span', {'class': "visually-hidden"})[-1]
    about = about.text.strip()

    return about


nlp = spacy.load("en_core_web_sm")


def tokeniser(sentence):

    sentence = re.sub("[]|||[]", " ", sentence)

    sentence = re.sub("/r/[0-9A-Za-z]", "", sentence)

    MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                  'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
                  'MBTI']
    MBTI_types = [ti.lower() for ti in MBTI_types] + \
        [ti.lower() + 's' for ti in MBTI_types]

    tokens = nlp(sentence)

    tokens = [ti for ti in tokens if ti.lower_ not in STOP_WORDS]
    tokens = [ti for ti in tokens if not ti.is_space]
    tokens = [ti for ti in tokens if not ti.is_punct]
    tokens = [ti for ti in tokens if not ti.like_num]
    tokens = [ti for ti in tokens if not ti.like_url]
    tokens = [ti for ti in tokens if not ti.like_email]
    tokens = [ti for ti in tokens if ti.lower_ not in MBTI_types]

    # lemmatize
    tokens = [ti.lemma_ for ti in tokens if ti.lemma_ not in STOP_WORDS]
    tokens = [ti for ti in tokens if len(ti) > 1]

    return tokens


def analyze(text):
    model_save_location = "./models/"
    # lambda function that returns the same data back
    dummy_fn = lambda x:x
    def dummy_fn(x):
        return x

    with open(model_save_location + 'cv.pickle', 'rb') as f:
        cv = pickle.load(f)
    with open(model_save_location + 'idf_transformer.pickle', 'rb') as f:
        idf_transformer = pickle.load(f)
    with open(model_save_location + 'LR_clf_IE_kaggle.pickle', 'rb') as f:
        lr_ie = pickle.load(f)
    with open(model_save_location + 'LR_clf_JP_kaggle.pickle', 'rb') as f:
        lr_jp = pickle.load(f)
    with open(model_save_location + 'LR_clf_NS_kaggle.pickle', 'rb') as f:
        lr_ns = pickle.load(f)
    with open(model_save_location + 'LR_clf_TF_kaggle.pickle', 'rb') as f:
        lr_tf = pickle.load(f)

    c = cv.transform([tokeniser(text)])
    x = idf_transformer.transform(c)

    ie = lr_ie.predict_proba(x).flatten()
    ns = lr_ns.predict_proba(x).flatten()
    tf = lr_tf.predict_proba(x).flatten()
    jp = lr_jp.predict_proba(x).flatten()

    probs = np.vstack([ie, ns, tf, jp])

    names = ["Introversion - Extroversion",
             "Intuiting - Sensing",
             "Thinking - Feeling",
             "Judging - Perceiving"]

    result = {}
    for i in range(len(names)):
        result[names[i]] = probs[i]
    return result


def parsePDF(location):
    text = []
    pdfFileObj = open(location, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(0)
        text.append(pageObj.extractText())
    pdfFileObj.close()
    return text


def analyzePDF(location):
    titles = ["Summary", "Summary of", "Summary of Experience", "about"]

    for page in parsePDF(location):
        lines = page.split("\n")
        lines = list(map(lambda x: x.strip(), lines))
        summary_title = ""
        title = ""
        content = ""
        breakdown = dict()
        for line in lines:
            if line.isupper():
                for t in titles:
                    if t.lower() in line.lower():
                        summary_title = line
                        break
                if content != "":
                    breakdown[title] = content
                    content = ""
                title = line
            else:
                if line != "":
                    content += line + " "

    return analyze(breakdown[summary_title].strip())
