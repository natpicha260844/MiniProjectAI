import tweepy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# กำหนดข้อมูลการเข้าถึง Twitter API และรับข้อมูล
consumer_key = 'Your_Consumer_Key'
consumer_secret = 'Your_Consumer_Secret'
access_token = 'Your_Access_Token'
access_token_secret = 'Your_Access_Token_Secret'

# สร้าง Tweepy API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# ดึงข้อมูลจาก Twitter
tweets = api.user_timeline(screen_name='twitterusername', count=10, tweet_mode='extended')

# สร้างโมเดล TensorFlow/Keras เพื่อวิเคราะห์ข้อความ
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=100),
    keras.layers.LSTM(units=64),
    keras.layers.Dense(1, activation='sigmoid')
])

# รับข้อความจาก tweets
texts = [tweet.full_text for tweet in tweets]

# ใช้ Tokenizer ในการเตรียมข้อมูลข้อความ
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')