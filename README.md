# NLP N-gram Language Modeling

### Requirements:
```
pip install nltk
```

### Running the code
```
python main.py
```

### Project Structure

- data_tokenization.py — Tokenization modes (word / whitespace / punct / nltk) plus file read/write operations.

- ngram_calc.py — N-gram counting and n-gram probability computation.

- perplexity.py — Perplexity calculation utilities.

- smoothing.py — Implementations of smoothing techniques (e.g., add-k/Laplace, Kneser–Ney, backoff/interpolation).

- unk_handling.py — Functions to build train vocab and map OOVs to <unk>.

- print_results.py — Helpers to format and print experiment results.



