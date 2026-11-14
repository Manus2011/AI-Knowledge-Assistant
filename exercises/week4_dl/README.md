week 4 notes

- digit_recognizer.py: numpy-only neural net with manual backprop, ~97% on the 8x8 digits test split (sklearn's MLP gets ~98%)
- cnn_digits.py: small cnn in pytorch on the same split, ~97.3%
- neural_text_classifier.py: logreg vs MLP on the document categories. on text built from phrases the model never saw, both land around 55%
- lstm_text_classifier.py: same test with an lstm. it gets 25% (chance for 4 classes) because the unseen phrases are mostly words outside its vocabulary. data is synthetic so dont read much into any of these
- categorization_service can use CATEGORIZER=mlp or CATEGORIZER=lstm (lstm needs torch, requirements-dl.txt)
