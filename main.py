import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure NLTK downloads are available
nltk.download('vader_lexicon')

# Load the dataset
file_path = 'influencers_data.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna(subset=['content', 'reactions', 'comments'])

# Fill NaN values in 'time_spent' with a default value
data['time_spent'] = data['time_spent'].fillna('0 days')

# Convert 'time_spent' to a numerical format
def convert_time_spent(time_str):
    if 'day' in time_str:
        return int(time_str.split()[0]) * 24
    if 'week' in time_str:
        return int(time_str.split()[0]) * 24 * 7
    if 'month' in time_str:
        return int(time_str.split()[0]) * 24 * 30
    if 'year' in time_str:
        return int(time_str.split()[0]) * 24 * 365
    return 0

data['time_spent_hours'] = data['time_spent'].apply(convert_time_spent)

# Feature engineering
data['word_count'] = data['content'].apply(lambda x: len(x.split()))
data['char_count'] = data['content'].apply(lambda x: len(x))

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
data['sentiment'] = data['content'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Prepare text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['content'])
sequences = tokenizer.texts_to_sequences(data['content'])
padded_sequences = pad_sequences(sequences, maxlen=100)

# Prepare features and labels
X_text = padded_sequences
X_meta = data[['word_count', 'char_count', 'sentiment', 'time_spent_hours']]
y = data[['reactions', 'comments']]

# Standardize metadata
scaler = StandardScaler()
X_meta = scaler.fit_transform(X_meta)

# Split the data
X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(X_text, X_meta, y, test_size=0.2, random_state=42)

# Text input branch
input_text = Input(shape=(100,))
embedding = Embedding(input_dim=5000, output_dim=128)(input_text)
conv = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding)
pool = GlobalMaxPooling1D()(conv)

# Metadata input branch
input_meta = Input(shape=(4,))
dense_meta = Dense(32, activation='relu')(input_meta)

# Concatenate branches
concatenated = concatenate([pool, dense_meta])
output = Dense(2, activation='linear')(concatenated)

# Create model
model = Model(inputs=[input_text, input_meta], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit([X_train_text, X_train_meta], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on test set
y_pred = model.predict([X_test_text, X_test_meta])

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Visualize the correlation matrix
correlation_matrix = data[['sentiment', 'reactions', 'comments']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot instead of showing it
