week 4 notes

- digit_recognizer.py: numpy-only neural net with manual backprop, ~97% on the 8x8 digits test split (sklearn's MLP gets ~98%)
- neural_text_classifier.py: logreg vs MLP on the document categories. on text built from phrases the model never saw, both land around 55%, so the MLP isnt really better here. the data is synthetic so dont read much into it
- categorization_service can use the MLP with CATEGORIZER=mlp
- no torch/CNN/LSTM, couldnt install torch in my environment
