from flask import Flask, render_template, request
import pickle
import re
from tfidf_vectorizer import TFIDFVectorizer


app = Flask(__name__)

with open("./model/vectorizer.pkl", "rb") as tfidf:
    vectorizer = pickle.load(tfidf)

with open("./model/classifier.pkl", "rb") as classifier:
    mnb = pickle.load(classifier)


def data_cleaning(string):
    text = re.sub(
        "\,|\@|\-|\"|'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—|\०|\१|\२|\३|\४|\५|\६|\७|\८|\९|[0-9]",
        "",
        string,
    )
    return text


def stop_word_remove(array_element):
    array_element_set = set(array_element)
    stop_words_file = open("../data/stopwords.txt", "r", encoding="utf-8")
    stop_words = stop_words_file.read()
    stop_words = stop_words.split("\n")
    final_list = list(array_element_set.difference(stop_words))
    return final_list


def predict_sentiment(sentence):
    # Preprocess the input sentence
    cleaned_sentence = data_cleaning(sentence)
    tokenized_sentence = cleaned_sentence.split(" ")
    stop_word_removed = stop_word_remove(tokenized_sentence)
    stop_words = [word for word in tokenized_sentence if word not in stop_word_removed]
    
    # Transform the preprocessed sentence using TF-IDF vectorizer
    sentence_features = vectorizer.transform([stop_word_removed])
    tf_idf = vectorizer.get_tf_idf_info(stop_word_removed,sentence_features)
    # Use the trained classifier to predict the sentiment label
    predicted_label = mnb.predict(sentence_features)
    
    print(tokenized_sentence)
    print(stop_word_removed)
    print(stop_words)
    print(tf_idf)
    
    return (
        cleaned_sentence,
        tokenized_sentence,
        stop_word_removed,
        stop_words,
        tf_idf,
        predicted_label[0],
    )  # Return the predicted sentiment label


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        # print(request.form)
        sentence = request.form["sentence"]
        (
            cleaned_sentence,
            tokenized_sentence,
            stop_word_removed,
            stop_words,
            tf_idf,
            predicted_sentiment,
        ) = predict_sentiment(sentence)

        # Map numerical labels to sentiment labels
        if predicted_sentiment == 1:
            sentiment_label = "Positive"
        elif predicted_sentiment == -1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        return render_template(
            "index.html",
            cleaned=cleaned_sentence,
            token=tokenized_sentence,
            stop_word_removed=stop_word_removed,
            stop_words=stop_words,
            tf_idf = tf_idf,
            sentiment=sentiment_label,
            sentence=sentence,
        )
    if request.method == "GET":
        return render_template("index.html")


@app.route(
    "/process/<sentence>/<cleaned>/<token>/<stop_word_removed>/<sentiment>/<tf_idf>/<stop_words>", methods=["GET"]
)
def process(sentence, cleaned, token, stop_word_removed, sentiment, tf_idf, stop_words):
    if request.method == "GET":
        # sentence = request.form["sentence"]
        # cleaned_sentence,tokenized_sentence,stop_word_removed,predicted_sentiment = predict_sentiment(sentence)

        return render_template(
            "process.html",
            cleaned=cleaned,
            token=token,
            stop_word_removed=stop_word_removed,
            sentiment=sentiment,
            sentence=sentence,
            tf_idf=tf_idf,
            stop_words=stop_words,
        )


if __name__ == "__main__":
    app.run(debug=True)
