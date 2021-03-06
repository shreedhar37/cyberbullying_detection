import asyncio
import hashlib
from wsgiref.util import request_uri
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
import easyocr
from werkzeug.utils import secure_filename
# import torchvision
from langdetect import detect
from flask_mail import Mail, Message
import datetime as dt
import joblib
from asyncio import new_event_loop, set_event_loop
from wordcloud import WordCloud

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





@app.route("/", methods=["GET", "POST"])
def login():
    try:
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

    except Exception as e:
        return render_template('index.html', msg = e)

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
    try:
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

        return render_template('register.html', msg = msg)
    
    except Exception as e:
        return render_template('register.html', msg = e)

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
 
    if "loggedin" in session:

        try:
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

            c.Limit = 50

            # c.Near = "India"

            c.Store_csv = True  # store tweets in a csv file

            c.Output = os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv"  # path to csv file

            asyncio.set_event_loop(asyncio.new_event_loop())
            
            twint.run.Search(c)
            

            test = pd.read_csv(
                os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv", 
                error_bad_lines=False
            )


            def preprocess(input_txt):
            
                try:
                
                    if detect(input_txt) == 'en':
                        pattern = "[^@[\w]+]| [a-zA-Z]+"
                        result = ''.join(re.findall(pattern, input_txt))
                        return result
                
                except Exception as e:
                    pass
            
            # clean the tweets and remove non-english tweets
            print("Length before preprocessing : ", len(test.tweet))
            
            test['cleaned_tweets'] = np.vectorize(preprocess)(test['tweet'])
            test ['cleaned_tweets'] = test['cleaned_tweets'].replace('None', np.nan)
            test = test.dropna(subset= ['cleaned_tweets'], how='all')
            test = test.reset_index(drop=True)
        
        
            test.drop_duplicates(subset=['cleaned_tweets'], inplace = True)
        

            print("Length after preprocessing : ", len(test.tweet))
            test.to_csv(os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv")
            test = pd.read_csv(
                os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv", 
                error_bad_lines=False
            )

            # NLP Techniques
            lemmatizer = WordNetLemmatizer()
            tweets = []
            for i in test['cleaned_tweets']:
                tweets.append(i)
            
            corpus = []
            
            for i in range(len(tweets)):
                tweet = re.sub('[^a-zA-Z]', ' ', tweets[i])
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
                tweet = " ".join(tweet)
                corpus.append(tweet)
            
            loaded_model = joblib.load("static\model\\model.pkl")
            
            loaded_vectorizer = joblib.load("static\model\\vectorizer.pkl")
            
            # tweets = loaded_vectorizer.transform(corpus)


            # prediction = loaded_model.predict(tweets)

            result = []
            label = []
            
            for i in range(len(test)):
                    
                text_transformed =  loaded_vectorizer.transform([corpus[i]])
                text_result = loaded_model.predict(text_transformed)

                if text_result[0] != 'none':
                    # print("inside text with index: ",i)
                    if text_result[0] == 'racism':
                        
                        label.append(1)
                    
                    if text_result[0] == 'sexism':
                        label.append(2)
                    
                    if text_result[0] == 'other':
                        label.append(3)

                    result.append(text_result[0])
                
            
                hashtags_result = ['none']
                hashtags = ""
                if len(test['hashtags'][i]) > 2:
        
                    text = test['hashtags'][i]
                    pattern = "[a-zA-Z]+"
                    hashtags = " ".join(re.findall(pattern, text))
                    # print(hashtags)
        
                    hashtags_transformed = loaded_vectorizer.transform([hashtags])
                    hashtags_result = loaded_model.predict(hashtags_transformed)
            

                    if ( text_result[0] == 'none' and hashtags_result[0] != 'none'):
                        # print("inside hashtags with index: ",i)
                        
                        print(hashtags_result[0])
                        if hashtags_result[0] == 'racism':
                            label.append(1)
                        
                        if hashtags_result[0] == 'sexism':
                            label.append(2)
                        
                        if hashtags_result[0] == 'other':
                            label.append(3)

                        result.append(hashtags_result[0])

                if (text_result[0] == 'none' and hashtags_result[0] == 'none'):
                    
                    result.append(text_result[0])
                    label.append(0)

        
            # result = []

            # label = []

            # for i in prediction:

            #     if i == 'none':
            #         label.append(0)
            #     elif i == 'racism':
            #         label.append(1)
            #     elif i == 'sexism' :
            #         label.append(2)
            #     else :
            #         label.append(3)
                
            #     result.append(i)
            print(len(test['tweet']), len(result), len(label))
            # visualization 
            bar= pd.DataFrame(list(zip(label, result)),columns =['label', 'category'])
            bar.groupby('category').label.value_counts().plot(kind = "bar", color = ["pink", "orange", "red", "yellow", "blue"])
            plt.xlabel("Category of tweet")
            plt.xticks(rotation='horizontal')
            plt.ylabel('Number of tweets')
            plt.title("Visualize numbers of Category of tweets")
            plt.savefig("static\graphs\\" + search_query + "_.png", bbox_inches='tight')
        
        # wordcloud

            all_words = ' '.join([text for text in test['tweet']])

            wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(all_words)
            wordcloud.to_file("static\wordclouds\\" + search_query + "_.png")

         
         
            return render_template(
                "home.html",
                search_query=search_query,
                Tweets=test.tweet.values,
                result=result
            )

        except FileNotFoundError:
            error = 'found 0 tweets in this search'
        
        except Exception as e:
            error = e
            
        
        return render_template(
                'home.html',
                error = error
            )

    else:
        return redirect(url_for("login"))
         

@app.route("/imageResult", methods=["GET", "POST"])
def imageResult():
    if "loggedin" in session:
    
        try:
            model_filename  = 'static\model\\model.pkl'
            vectorizer_filename= 'static\model\\vectorizer.pkl'
            
            loaded_vectorizer = joblib.load(vectorizer_filename) # load vectorizer
            loaded_model = joblib.load(model_filename) # load model
            
            search_query = str(request.form["text"])
            
            
            c = twint.Config()

            c.Search = search_query.split(" ")

            c.Lang = "en"

            c.Min_likes = 100

            c.Limit = 10

            # c.Near = "India"

            c.Store_csv = True  # store tweets in a csv file

            c.Output = os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv"  # path to csv file

            asyncio.set_event_loop(asyncio.new_event_loop())
            
            twint.run.Search(c)
            

            test = pd.read_csv(
                os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv", 
                error_bad_lines=False
            )
            
            def preprocess(input_txt):
                
                try:
                
                    if detect(input_txt) == 'en':
                        pattern = "[^@[\w]+]| [a-zA-Z]+"
                        result = ''.join(re.findall(pattern, input_txt))
                        return result
                
                except Exception as e:
                    pass
            
            # clean the tweets and remove non-english tweets
            print("Length before preprocessing : ", len(test.tweet))
            
            test['cleaned_tweets'] = np.vectorize(preprocess)(test['tweet'])
            test ['cleaned_tweets'] = test['cleaned_tweets'].replace('None', np.nan)
            test = test.dropna(subset= ['cleaned_tweets'], how='all')
            test = test.reset_index(drop=True)
        
        
            test.drop_duplicates(subset=['cleaned_tweets'], inplace = True)
        

            print("Length after preprocessing : ", len(test.tweet))
            test.to_csv(os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv")
            test = pd.read_csv(
                os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv", 
                error_bad_lines=False
            )
            
            # NLP Techniques
            lemmatizer = WordNetLemmatizer()
            tweets = []
            for i in test['cleaned_tweets']:
                tweets.append(i)
            
            corpus = []
            
            for i in range(len(tweets)):
                tweet = re.sub('[^a-zA-Z]', ' ', tweets[i])
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
                tweet = " ".join(tweet)
                corpus.append(tweet)

            # update the record
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            query = 'INSERT INTO SEARCH_LOGS(UID, SEARCH_QUERY) VALUE(% s, % s)'
            cursor.execute(query, (session['UID'], search_query))
            mysql.connection.commit()
            
            def imgtotext(link):
        
                reader = easyocr.Reader(['en'], gpu = False)
    
               
                result = reader.readtext(link, paragraph='False', detail  = 0)
    
                img_corpus = []
                lemmatizer = WordNetLemmatizer()
                
                for i in range(len(result)):
                    tweet = re.sub('[^a-zA-Z]', ' ', result[i])
                    tweet = tweet.lower()
                    tweet = tweet.split()
                    tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
                    tweet = " ".join(tweet)
                    img_corpus.append(tweet)

                text  = " "
                text = text.join(img_corpus)
                # print(text)
                text_transformed = loaded_vectorizer.transform([text])

                
                return loaded_model.predict(text_transformed)

            text_result = ""
            hashtags_result = ""
            links = []
            index = []
            imgLinks = []
            result = []
            
            for i in range(len(test)):
                
        
                if len(test['photos'][i]) > 2:
                    
                    text_transformed =  loaded_vectorizer.transform([corpus[i]])
                    text_result = loaded_model.predict(text_transformed)
                    print("Text result: ",text_result)
                    text = test['photos'][i]
                    pattern = "https:[^']+"
                    links = re.findall(pattern, text)
                    # print(links)
            
                    if (text_result) == 'none':
                        # print("inside hashtag condition")
                        text = test['hashtags'][i]
                        pattern = "\w+"
                        hashtags = " ".join(re.findall(pattern, text))
                        # print(hashtags)
                        hashtags_transformed = loaded_vectorizer.transform([hashtags])
                        hashtags_result = loaded_model.predict(hashtags_transformed)
                        # print("Hashtag result: " ,hashtags_result[0])
               
                        if hashtags_result[0] == 'none':
                            print("Inside img condition")
                            if len(links) != 0:
                               # for link in links:
                                img_result = imgtotext(links[0])
                                print((img_result))
                                if (img_result) != 'none':
                                    imgLinks.append(links[0])
                                    index.append(i)
                                    result.append(img_result)
                            
                            
                        else:
                            index.append(i)
                            imgLinks.append(links[0])
                            result.append(hashtags_result)
                            print("\n hashtag Prediction: ", hashtags_result)
            
                    else:
                        index.append(i)
                        imgLinks.append(links[0])
                        result.append(text_result)
                        print("\n text Prediction: ", text_result)
                    
            print(index)
            print(result)
            print(imgLinks)
            return render_template(
                "image.html",
                search_query=search_query,
                index = index,
                imgLinks = imgLinks,
                result=result,
                tweets = test
            )


        except FileNotFoundError:
            error = 'found 0 tweets in this search'
        
        except Exception as e:
            error = e
            
        
        return render_template(
                'image.html',
                error = error
            )
    else:
        return redirect(url_for("login"))
        


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
    if "loggedin" in session:
        try:

            search_query = request.form['search_query']
            index = int(request.form['i'])
            test = pd.read_csv(
                os.getcwd() + "\static\scraped_tweets\\" + search_query + ".csv", error_bad_lines=False
            )

            tweet_link = test['link'][index]
            tweet_category = request.form['result[i]']
            

            
            
            return render_template('send_report.html', 
                                    tweet_link= tweet_link,    
                                    tweet_category = tweet_category
                                    )

        except Exception as e:
            return render_template('send_report.html', msg = e)
    
    else:
        return redirect(url_for("login"))


@app.route('/success', methods=['GET', 'POST'])
def success():
    if "loggedin" in session:
        try:
            with app.app_context():
                msg = Message(
                    subject= "Twitter Cyberbullying",
                    sender=app.config.get("MAIL_USERNAME"),
                    recipients=[request.form['email']],
                    body=request.form['msg'])

            mail.send(msg)

            # find the record
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            query = 'SELECT REPORT_COUNT FROM USER WHERE UID = %s'
            cursor.execute(query, (session['UID'],))
            account = cursor.fetchone()

            # update the record
            query = 'UPDATE USER SET REPORT_COUNT = %s WHERE UID = %s'
            cursor.execute(query, (account['REPORT_COUNT'] + 1, session['UID']))
            
            query = 'INSERT INTO USER_LOGS(UID, TWEET_LINK, TWEET_CATEGORY) VALUE(% s, % s, % s)'
            tweet_link = request.form['tweet_link']
            tweet_category = request.form['tweet_category']
            cursor.execute(query, (session['UID'], tweet_link, tweet_category))
            
            mysql.connection.commit()

            msg = 'Your report has been sent to Cybercrime Branch. Thank you for reporting!!'
            return render_template('home.html', success = msg)

        except Exception as e:
            print(e)
            return render_template('home.html', error=e)

    else:
        return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=False, port=5000)