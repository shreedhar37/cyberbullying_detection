import asyncio
import hashlib
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import twint
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import os
# from PIL import Image
# import easyocr
from werkzeug.utils import secure_filename
# import torchvision
import preprocessor as p
from langdetect import detect
from flask_mail import Mail, Message
import datetime as dt


app = Flask(__name__)
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": '',
    "MAIL_PASSWORD": ''
}

app.config.update(mail_settings)
mail = Mail(app)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = "your secret key"


# File upload configuration
app.config["UPLOAD_FOLDER"] = "/flask"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'project'


# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/ - this will be the login page, we need to use both GET and POST requests


# Model building
dataset = pd.read_csv(
    "D:\Programming\BE PROJECT\datasets\\bullying_dataset.csv")


# # data preprocessing

# def convert_lower(text):
#     return text.lower()

# dataset['tweet'] = dataset['tweet'].apply(convert_lower)


# def remove_stopwords(text):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text)
#     return [x for x in words if x not in stop_words]

# dataset['tweet'] = dataset['tweet'].apply(remove_stopwords)

# def lemmatize_word(text):
#     wordnet = WordNetLemmatizer()
#     return " ".join([wordnet.lemmatize(word) for word in text])

# dataset['tweet'] = dataset['tweet'].apply(lemmatize_word)

x = dataset['tweet']
y = dataset['category']

x = np.array(dataset.iloc[:, 0].values)
y = np.array(dataset.category.values)
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(dataset.tweet).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True)

# create list of model and accuracy dicts
perform_list = []


def run_model(model_name, est_c, est_pnlty):

    mdl = ""

    if model_name == 'Random Forest':

        mdl = RandomForestClassifier(n_estimators=100, criterion='entropy')

    elif model_name == 'Multinomial Naive Bayes':

        mdl = MultinomialNB(alpha=1.0, fit_prior=True)

    elif model_name == 'Support Vector Classifer':

        mdl = SVC()

    elif model_name == 'Decision Tree Classifier':

        mdl = DecisionTreeClassifier()

    elif model_name == 'K Nearest Neighbour':

        mdl = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=4)

    elif model_name == 'BernoulliNB':

        mdl = BernoulliNB()

    oneVsRest = OneVsRestClassifier(mdl)

    oneVsRest.fit(x_train, y_train)

    y_pred = oneVsRest.predict(x_test)

    # Performance metrics

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Get precision, recall, f1 scores

    precision, recall, f1score, support = score(
        y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {model_name}: {accuracy} %')

    print(f'Precision : {precision}')

    print(f'Recall : {recall}')

    print(f'F1-score : {f1score}')

    # Add performance parameters to list

    perform_list.append(dict([

        ('Model', model_name),

        ('Test Accuracy', round(accuracy, 2)),

        ('Precision', round(precision, 2)),

        ('Recall', round(recall, 2)),

        ('F1', round(f1score, 2))

    ]))

run_model('BernoulliNB', est_c=None, est_pnlty=None)

# run_model('Multinomial Naive Bayes', est_c=None, est_pnlty=None)

# run_model('Support Vector Classifer', est_c=None, est_pnlty=None)
#run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)

# run_model('Decision Tree Classifier', est_c=None, est_pnlty=None)


# run_model('Random Forest', est_c=None, est_pnlty=None)

model_performance = pd.DataFrame(data=perform_list)
model_performance = model_performance[[
    'Model', 'Test Accuracy', 'Precision', 'Recall', 'F1']]
model_performance
model = model_performance["Model"]
max_value = model_performance["Test Accuracy"].max()
# print("The best accuracy of model is", max_value, "%")

classifier = RandomForestClassifier(
    n_estimators=100, criterion='entropy', random_state=0).fit(x_train, y_train)
# y_pred1 = cv.transform(['nigga got no chill', 'wassup bitches'])


@app.route("/", methods=["GET", "POST"])
def login():
    msg = ''
    if request.method == 'POST' and 'EMAIL' in request.form and 'PASSWORD' in request.form:
        EMAIL = request.form['EMAIL']
        PASSWORD = request.form['PASSWORD']
        PASSWORD = hashlib.md5(PASSWORD.encode())
        PASSWORD = PASSWORD.hexdigest()
       
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        query = 'SELECT * FROM USER WHERE EMAIL = % s AND PASSWORD = % s'
        cursor.execute(query, (EMAIL, PASSWORD,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['UID'] = account['UID']
            session['NAME'] = account['NAME']
            LOGIN_COUNT = account['LOGIN_COUNT'] + 1
            LAST_LOGIN = dt.datetime.now()

            query = 'UPDATE USER SET LOGIN_COUNT = % s, LAST_LOGIN = % s WHERE UID = % s'

            cursor.execute(query, (LOGIN_COUNT, LAST_LOGIN, account['UID'],))

            mysql.connection.commit()

            return render_template('home.html')
        else:
            msg = 'Incorrect username / password !'

    return render_template('index.html', msg=msg)


# http://localhost:5000/logout - this will be the logout page


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('UID', None)
    session.pop('NAME', None)
    return redirect(url_for('login'))


# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests

# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests
@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ''
    if request.method == 'POST' and 'NAME' in request.form and 'EMAIL' in request.form and 'PASSWORD' in request.form and 'CPASSWORD' in request.form:
        NAME = request.form['NAME']
        EMAIL = request.form['EMAIL']
        PASSWORD = request.form['PASSWORD']
        CPASSWORD = request.form['CPASSWORD']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        query = 'SELECT EMAIL FROM USER WHERE EMAIL = % s'
        cursor.execute(query, (EMAIL, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', EMAIL):
            msg = 'Invalid email address !'

        elif not NAME or not EMAIL or not PASSWORD or not CPASSWORD:
            msg = 'Please fill out the form !'
        elif PASSWORD != CPASSWORD:
            msg = "Confirm Password doesn't match with password !!"

        else:
            PASSWORD = hashlib.md5(PASSWORD.encode())
            PASSWORD = PASSWORD.hexdigest()
            query = 'INSERT INTO USER(NAME, EMAIL, PASSWORD) VALUES(%s, %s, %s)'
            cursor.execute(query, (NAME, EMAIL, PASSWORD,))
            mysql.connection.commit()
            msg = 'You have successfully registered ! Please Login with your credentials'
            return render_template('index.html', msg=msg)

    elif request.method == 'POST':
        msg = 'Please fill out the form ! '

    return render_template('register.html', msg=msg)

# http://localhost:5000/home - this will be the home page, only accessible for loggedin users


@app.route("/home")
def home():
    # Check if user is loggedin
    if "loggedin" in session:
        # User is loggedin show them the home page
        return render_template("home.html", username=session['NAME'])
    # User is not loggedin redirect to login page
    return redirect(url_for("login"))


# http://localhost:5000/profile - this will be the profile page, only accessible for loggedin users
@app.route("/profile")
def profile():
    # Check if user is loggedin
    if "loggedin" in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        query = 'SELECT * FROM USER WHERE UID = % s'
        cursor.execute(query, (session['UID'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template("profile.html", account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for("login"))


@app.route("/home")
def my_form():
    return render_template("home.html")


@app.route("/home", methods=["POST"])
def my_form_post():

    search_query = str(request.form["text"])

    # update the record
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    query = 'INSERT INTO SEARCH_LOGS(UID, SEARCH_QUERY) VALUE(% s, % s)'
    cursor.execute(query, (session['UID'], search_query))
    mysql.connection.commit()
    c = twint.Config()

    c.Search = search_query.split(" ")

    c.Lang = "en"

    c.Min_likes = 100

    c.Limit = 250

    # c.Near = "India"

    c.Store_csv = True  # store tweets in a csv file

    c.Output = os.getcwd() + search_query + ".csv"  # path to csv file

    asyncio.set_event_loop(asyncio.new_event_loop())

    twint.run.Search(c)

    test = pd.read_csv(
        os.getcwd() + search_query + ".csv", error_bad_lines=False
    )

    # data preprocessing
    # test['tweet'] = test['tweet'].apply(convert_lower)

    # test['tweet'] = test['tweet'].apply(remove_stopwords)

    # test['tweet'] = test['tweet'].apply(lemmatize_word)

    tweets = test['tweet'][:20].values

    prediction = cv.transform(tweets)

    prediction = classifier.predict(prediction)

    print(prediction)

    result = []

    for i in prediction:

        result.append(i)

    # print(result)
    # return render_template('home.html')

    return render_template(
        "home.html",
        search_query=search_query,
        Tweets=tweets,
        result=result
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":

        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads

        basepath = os.getcwd()

        file_path = os.path.join(
            basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # reader = easyocr.Reader(['en'], gpu = False)
        # img_txt = reader.readtext(file_path, paragraph="False", detail = 0)

        text = " "

        # text = text.join(img_txt)

        # text_transformed = cv.transform([text])

        # result = classifier.predict(text_transformed)

        # return render_template("image.html", Text = text, result = result)


@app.route("/image")
def image():
    # Check if user is loggedin
    if "loggedin" in session:
        # User is loggedin show them the home page
        return render_template("image.html")
    # User is not loggedin redirect to login page
    return redirect(url_for("login"))


@app.route('/send_report', methods=['GET', 'POST'])
def send_report():
    search_query = request.form['search_query']
    index = int(request.form['i'])
    test = pd.read_csv(
        os.getcwd() + search_query + ".csv", error_bad_lines=False
    )

    tweet_id = test['id'][index]
    tweet_username = test['username'][index]
    tweet_owner = test['name'][index]
    tweet = test['tweet'][index]

    

    # update the record
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    query = 'INSERT INTO USER_LOGS(UID, TWITTER_USERNAME, TWEET) VALUE(% s, % s, % s)'
    cursor.execute(query, (session['UID'], tweet_username, tweet))
    mysql.connection.commit()
    
    return render_template('send_report.html', tweet_id=tweet_id, tweet_username=tweet_username, tweet_owner=tweet_owner, tweet=tweet)


@app.route('/success', methods=['GET', 'POST'])
def success():
    try:
        with app.app_context():
            msg = Message(
                subject=request.form['subject'],
                sender=app.config.get("MAIL_USERNAME"),
                recipients=[request.form['email']],
                body=request.form['msg'])

        mail.send(msg)

        # find the record
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        query = 'SELECT REPORT_COUNT FROM USER WHERE UID = % s'
        cursor = cursor.execute(query, (session['UID']))
        account = cursor.fetchone()

        # update the record
        query = 'UPDATE USER SET REPORT_COUNT = % s WHERE UID = % s'
        report_count = account['REPORT_COUNT'] + 1
        cursor.execute(query, (report_count, session['UID']))
        mysql.connection.commit()

        return render_template('success.html')

    except Exception as e:
        print(e)
        return render_template('success.html', error=e)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
