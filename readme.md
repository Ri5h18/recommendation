# MovieLens Recommendation System

This project builds a movie recommendation system using the MovieLens dataset (latest small version). It leverages TensorFlow and Keras to train a neural network model that predicts movie ratings for users based on their past interactions.

---

## Overview

* **Dataset**: MovieLens latest small dataset (latest 100k+ ratings)
* **Goal**: Predict movie ratings and recommend movies a user might like
* **Model**: Neural network with user and movie embeddings, trained with binary cross-entropy loss
* **Evaluation**: Training and validation loss over epochs, top movie recommendations per user

---

## Setup & Usage

1. **Download & Extract Data**

   The script automatically downloads and extracts the MovieLens dataset:

   ```python
   http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   ```

2. **Data Preprocessing**

   * Maps raw `userId` and `movieId` to encoded integers
   * Normalizes ratings between 0 and 1 for easier training

3. **Model Architecture**

   * Two embedding layers: one for users, one for movies
   * Bias terms for users and movies
   * Dot product of embeddings plus biases passed through a sigmoid activation to predict normalized ratings

4. **Training**

   * 5 epochs using Adam optimizer and binary cross-entropy loss
   * 90/10 train-validation split

5. **Recommendations**

   * For a given user, the model predicts ratings on movies they haven't watched
   * Top 10 movies with highest predicted ratings are recommended

---

## Code Execution

Run the script as-is to:

* Download and preprocess data
* Train the recommender model
* Plot training and validation loss
* Display top-rated movies for a random user and recommend new movies

---

## Output Example

```plaintext
Showing recommendations for user: 337
====================================
Movies with high ratings from user
--------------------------------
Shawshank Redemption, The (1994) : Crime|Drama
Craft, The (1996) : Drama|Fantasy|Horror|Thriller
...

Top 10 movie recommendations
--------------------------------
Lone Star (1996) : Drama|Mystery|Western
Emma (1996) : Comedy|Drama|Romance
Star Wars: Episode V - The Empire Strikes Back (1980) : Action|Adventure|Sci-Fi
...
```

---

## Limitations & Notes

* The model uses binary cross-entropy on normalized ratings, which may not be ideal for explicit rating prediction. Consider using regression losses (MSE) or ranking losses.
* Only 5 epochs of training—more epochs or hyperparameter tuning could improve recommendations.
* The dataset is relatively small; results may not generalize well to larger or different datasets.
* Cold-start problem not addressed: new users or movies without prior ratings won't get meaningful embeddings.

---

## Requirements

* Python 3.7+
* TensorFlow 2.x
* Pandas
* NumPy
* Matplotlib

---

## References

* [MovieLens dataset](https://grouplens.org/datasets/movielens/)
* [TensorFlow Keras Embeddings](https://www.tensorflow.org/tutorials/structured_data/collaborative_filtering)

---
