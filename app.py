import asyncio
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import preprocessor as p
from langdetect import detect
import twint
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pytesseract
import cv2 as CV
from werkzeug.utils import secure_filename
from PIL import Image
import preprocessor as p
from langdetect import detect 

from flask_mail import Mail, Message
# tesseract config
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/tesseract.exe"
tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/tessdata"'


app = Flask(__name__)
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": 'your email goes here',
    "MAIL_PASSWORD": 'your password goes here'
}

app.config.update(mail_settings)
mail = Mail(app)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = "your secret key"


# File upload configuration
app.config["UPLOAD_FOLDER"] = "/flask"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Enter your database connection details below
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "pythonlogin"

# Intialize MySQL
mysql = MySQL(app)
# http://localhost:5000/ - this will be the login page, we need to use both GET and POST requests



# Model building
train = pd.read_csv(
        "D:\\Programming\\BE PROJECT\\bullying_dataset.csv", error_bad_lines=False
    )
train.drop_duplicates(keep=False, inplace=True)


# test = pd.read_csv("bullying_dataset.csv")

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
        train.tweet, train.label, test_size=0.20
)
cv = CountVectorizer(lowercase=False)
features = cv.fit_transform(X_train)
    # build a model

tunned_parameters = {
        "kernel": ["linear", "rbf"],
        "gamma": [1e-3, 1e-4],
        "C": [1, 10, 100, 1000],
    }

svm_model = GridSearchCV(svm.SVC(), tunned_parameters)

svm_model.fit(features, y_train)

features_test = cv.transform(X_test)
svm_model_score = svm_model.score(features_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(
        train.tweet, train.label, test_size=0.20
    )
v = CountVectorizer(lowercase=False)
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_count, y_train)
X_test_count = v.transform(X_test)
naive_bayes_model_score = naive_bayes_model.score(X_test_count, y_test)


if naive_bayes_model_score > svm_model_score:
    model = naive_bayes_model
else:
    model = svm_model

@app.route("/", methods=["GET", "POST"])
def login():
    # Output message if something goes wrong...
    msg = ""
    # Check if "username" and "password" POST requests exist (user submitted form)
    if (
        request.method == "POST"
        and "username" in request.form
        and "password" in request.form
    ):
        # Create variables for easy access
        username = request.form["username"]
        password = request.form["password"]
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT * FROM accounts WHERE username = %s AND password = %s",
            (
                username,
                password,
            ),
        )
        account = cursor.fetchone()
        if account:
            # Create session data, we can access this data in other routes
            session["loggedin"] = True
            session["id"] = account["id"]
            session["username"] = account["username"]
            # Redirect to home page
            return redirect(url_for("home"))
        else:
            # Account doesnt exist or username/password incorrect
            msg = "Incorrect username/password!"
    return render_template("index.html", msg=msg)


# http://localhost:5000/logout - this will be the logout page


@app.route("/logout")
def logout():
    # Remove session data, this will log the user out
    session.pop("loggedin", None)
    session.pop("id", None)
    session.pop("username", None)
    # Redirect to login page
    return redirect(url_for("login"))


# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests

# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests
@app.route("/register", methods=["GET", "POST"])
def register():
    # Output message if something goes wrong...
    msg = ""
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if (
        request.method == "POST"
        and "username" in request.form
        and "password" in request.form
        and "email" in request.form
    ):
        # Create variables for easy access
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM accounts WHERE username = %s", (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = "Account already exists!"
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            msg = "Invalid email address!"
        elif not re.match(r"[A-Za-z0-9]+", username):
            msg = "Username must contain only characters and numbers!"
        elif not username or not password or not email:
            msg = "Please fill out the form!"
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute(
                "INSERT INTO accounts VALUES (NULL, %s, %s, %s)",
                (
                    username,
                    password,
                    email,
                ),
            )
            mysql.connection.commit()
            msg = "You have successfully registered!"
    elif request.method == "POST":
        # Form is empty... (no POST data)
        msg = "Please fill out the form!"
    # Show registration form with message (if any)
    return render_template("register.html", msg=msg)


# http://localhost:5000/home - this will be the home page, only accessible for loggedin users
@app.route("/home")
def home():
    # Check if user is loggedin
    if "loggedin" in session:
        # User is loggedin show them the home page
        return render_template("home.html", username=session["username"])
    # User is not loggedin redirect to login page
    return redirect(url_for("login"))


# http://localhost:5000/profile - this will be the profile page, only accessible for loggedin users
@app.route("/profile")
def profile():
    # Check if user is loggedin
    if "loggedin" in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM accounts WHERE id = %s", (session["id"],))
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
    
    train = pd.read_csv(
        "D:\\Programming\\BE PROJECT\\bullying_dataset.csv", error_bad_lines=False
    )
    train.drop_duplicates(keep=False, inplace=True)

    
    topic = request.form["text"]
    

    c = twint.Config()
    c.Lang = "en"

    c.Min_likes = 25

    c.Limit = 5

    c.Near = "India"

    c.Store_csv = True  # store tweets in a csv file
    
    c.Output = os.getcwd() + topic + ".csv"  # path to csv file
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    twint.run.Search(c)
    
    test = pd.read_csv(
        os.getcwd() + topic + ".csv", error_bad_lines=False
    )

    # preprocessing and removing  non english tweets
    def preprocess(input_txt):
    
        try:
        
            if detect(input_txt) == 'en' and (len(p.clean(input_txt)) > 3):
            
                return p.clean(input_txt)

        except:
            pass

    # clean the tweets and remove non-english tweets

    test['tweet'] = np.vectorize(preprocess)(test['tweet'])

    # remove duplicates

    test.drop_duplicates( )

    # remove None values

    index = test[test['tweet'] == "None"].index

    test.drop(index, inplace = True)


    test['tweet'] = test['tweet'].head().apply(str)

    tweets = test['tweet'].head()
    tweets_count = cv.transform(tweets)

    return render_template(
        "home.html", topic = topic, Test=test, Tweets=tweets, Tweets_count=tweets_count, Model=model
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.getcwd()
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        img = CV.imread(file_path)

        d = pytesseract.image_to_string(img, config=tessdata_dir_config)

        return render_template("image.html", op=d)


@app.route("/image")
def image():
    return render_template("image.html")


@app.route('/send_report', methods = ['GET','POST'])
def send_report():
    topic = request.form['topic']
    index = int(request.form['i'])
    test = pd.read_csv(
        os.getcwd() + topic + ".csv", error_bad_lines=False
    )
    tweet_id = test['id'][index]
    tweet_username = test['username'][index]
    tweet_owner = test['name'][index]
    tweet = test['tweet'][index]
    return render_template('send_report.html', tweet_id = tweet_id, tweet_username = tweet_username, tweet_owner= tweet_owner, tweet = tweet)




@app.route('/success', methods = ['GET','POST'])
def success():
    try:
        with app.app_context():
            msg = Message(
                subject= request.form['subject'],
                sender=app.config.get("MAIL_USERNAME"),
                recipients=[request.form['email']],
                body= request.form['msg'])
            
        mail.send(msg)
        return render_template('success.html')

    except Exception as e:
        print(e)
        return render_template('success.html', error = e)

    


if __name__ == "__main__":
    app.run(debug=True, port = 5000)
