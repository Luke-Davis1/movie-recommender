from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

test_text = ["marvel superhero marvel", "superhero marvel superhero"]

cv = CountVectorizer()

# Get the counts of each word in every sentence
count_matrix = cv.fit_transform(test_text)

similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)