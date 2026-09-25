# textstat

```python
import textstat

test_data = (
    "Playing games has always been thought to be important to "
    "the development of well-balanced and creative children; "
    "however, what part, if any, they should play in the lives "
    "of adults has never been researched that deeply. I believe "
    "that playing games is every bit as important for adults "
    "as for children. Not only is taking time out to play games "
    "with our children and other adults valuable to building "
    "interpersonal relationships but is also a wonderful way "
    "to release built up tension."
)

textstat.flesch_reading_ease(test_data)
textstat.flesch_kincaid_grade(test_data)
textstat.smog_index(test_data)
textstat.coleman_liau_index(test_data)
textstat.automated_readability_index(test_data)
textstat.dale_chall_readability_score(test_data)
textstat.difficult_words(test_data)
textstat.linsear_write_formula(test_data)
textstat.gunning_fog(test_data)
textstat.text_standard(test_data)
textstat.fernandez_huerta(test_data)
textstat.szigriszt_pazos(test_data)
textstat.gutierrez_polini(test_data)
textstat.crawford(test_data)
textstat.gulpease_index(test_data)
textstat.osman(test_data)
```

## List of Functions

### Formulas

#### The Flesch Reading Ease formula

`textstat.flesch_reading_ease(text)`

Returns the Flesch Reading Ease Score.

The following table can be helpful to assess the ease of readability in a document.

The table is an example of values. While the maximum score is 121.22, there is no limit on how low the score can be. A negative score is valid.

Score Difficulty
90-100 Very Easy
80-89 Easy
70-79 Fairly Easy
60-69 Standard
50-59 Fairly Difficult
30-49 Difficult
0-29 Very Confusing

#### The Flesch-Kincaid Grade Level

`textstat.flesch_kincaid_grade(text)`

Returns the Flesch-Kincaid Grade of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.

#### The Fog Scale (Gunning FOG Formula)

`textstat.gunning_fog(text)`

Returns the FOG index of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.

#### The SMOG Index

`textstat.smog_index(text)`

Returns the SMOG index of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.

Texts of fewer than 30 sentences are statistically invalid, because the SMOG formula was normed on 30-sentence samples. textstat requires at least 3 sentences for a result.

#### Automated Readability Index

`textstat.automated_readability_index(text)`

Returns the ARI (Automated Readability Index) which outputs a number that approximates the grade level needed to comprehend the text.

For example if the ARI is 6.5, then the grade level to comprehend the text is 6th to 7th grade.

#### The Coleman-Liau Index

`textstat.coleman_liau_index(text)`

Returns the grade level of the text using the Coleman-Liau Formula. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.

#### Linsear Write Formula

`textstat.linsear_write_formula(text)`

Returns the grade level using the Linsear Write Formula. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.

#### Dale-Chall Readability Score

`textstat.dale_chall_readability_score(text)`

Different from other tests, since it uses a lookup table of the most commonly used 3000 English words. Thus it returns the grade level using the New

Dale-Chall Formula.

Score Understood by
4.9 or lower average 4th-grade student or lower
5.0–5.9 average 5th or 6th-grade student
6.0–6.9 average 7th or 8th-grade student
7.0–7.9 average 9th or 10th-grade student
8.0–8.9 average 11th or 12th-grade student
9.0–9.9 average 13th to 15th-grade (college) student

#### Readability Consensus based upon all the above tests

`textstat.text_standard(text, float_output=False)`

Based upon all the above tests, returns the estimated school grade level required to understand the text.

Optional float_output allows the score to be returned as a float. Defaults to False.

#### Spache Readability Formula

`textstat.spache_readability(text)`

Returns grade level of english text.

Intended for text written for children up to grade four.

#### Reading Time

`textstat.reading_time(text, ms_per_char=14.69)`

Returns the reading time of the given text.

Assumes 14.69ms per character.

### Aggregates and Averages

#### Syllable Count

`textstat.syllable_count(text)`

Returns the number of syllables present in the given text.

Uses the Python module Pyphen for syllable calculation in most languages, but defaults to cmudict for en_US.

#### Lexicon Count

`textstat.lexicon_count(text, removepunct=True)`

Calculates the number of words present in the text. Optional removepunct specifies whether we need to take punctuation symbols into account while counting lexicons. Default value is True, which removes the punctuation before counting lexicon items.

#### Sentence Count

`textstat.sentence_count(text)`

Returns the number of sentences present in the given text.

#### Character Count

`textstat.char_count(text, ignore_spaces=True)`

Returns the number of characters present in the given text.

#### Letter Count

`textstat.letter_count(text, ignore_spaces=True)`

Returns the number of characters present in the given text without punctuation.

#### Polysyllable Count

`textstat.polysyllabcount(text)`

Returns the number of words with a syllable count greater than or equal to 3.

#### Monosyllable Count

`textstat.monosyllabcount(text)`

Returns the number of words with a syllable count equal to one.
