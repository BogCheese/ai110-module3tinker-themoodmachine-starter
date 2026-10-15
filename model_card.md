# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I built and compared both models. The rule based model is the main one; the ML model is a comparison.

**Intended purpose:**
Classify short, social-media-style posts as `positive`, `negative`, `neutral`, or `mixed`. This is a classroom exercise, not a tool for judging real people's wellbeing.

**How it works (brief):**
- *Rule based:* lowercase and tokenize the text, count positive and negative word hits (negation flips a hit), then map the counts to a label.
- *ML:* turn each post into word counts (`CountVectorizer`) and fit a logistic regression on `SAMPLE_POSTS` and `TRUE_LABELS`.

## 2. Data

**Dataset description:**
14 posts in `SAMPLE_POSTS`: the 6 starter posts plus 8 I added with paired `append` calls (one post, then its label on the next line). An `assert` at the end of `dataset.py` fails if the two lists ever differ in length. Label counts: 4 positive, 5 negative, 3 mixed, 2 neutral.

**Labeling process:**
I labeled each post by what the writer most likely feels, not by which words appear. Hard or debatable ones:
- "Great, my phone died right before the big game :)" is `negative` (sarcasm), but the smiley makes it arguably mixed or positive.
- "Failed my exam 💀" is `negative`, but 💀 is often used for laughing, so someone else could read it as darkly funny.
- "Whatever, it's just another Monday" is `neutral`, but could be read as mildly negative.
- "Moving away for college, so excited but I'm gonna miss everyone 🥲" is `mixed`.

I labeled all of them myself, so there was no second opinion to expose disagreement.

**Important characteristics of your dataset:**
- Slang ("lowkey", "highkey", "no cap") and emoji (😂 💀 🥲 and `:)`)
- Sarcasm (traffic, dead phone)
- Mixed feelings (3 posts)
- Short, ambiguous posts ("This is fine")

**Possible issues with the dataset:**
- Tiny: 14 posts, and only 2 neutral ones.
- One labeler, no agreement check.
- No test set; every score below on `dataset.py` is training accuracy.
- The posts are mostly standard English with a sprinkling of slang I chose myself, so it does not represent real informal, regional, or non-standard writing.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- **Preprocess:** lowercase; a regex keeps words (with apostrophes, so `don't` stays whole), emoticons like `:)` and emoji as their own tokens; other punctuation is dropped.
- **Score:** each positive word is +1 and each negative word is −1.
- **Enhancement, negation:** a word in `NEGATORS` ("not", "never", "don't", ...) flips the next positive or negative word within two tokens, so "not happy" counts as negative. "no cap" is skipped because it is slang for "for real", not a negation.
- **Label:** score above 0 is `positive`, below 0 is `negative`, 0 is `neutral`, with one override. If a post has both positive and negative hits, it is `mixed` when it contains a contrast word ("but", "though", "although", "yet", "however"), and `negative` otherwise (a guess that praise next to a bad event is sarcasm).
- **Word lists:** the starter lists plus words I added after seeing failures. Positive: hopeful, proud, best, obsessed, 😂, 😊, `:)`. Negative: failed, died, stuck, miss, exhausted, 😢, `:(`.

**Strengths of this approach:**
- Every decision can be traced to tokens and a score, so wrong answers are easy to debug.
- Negation works: "I am not happy" is negative, and "im not happy" is too.
- Plain statements with listed words are reliable ("I love this class so much", "Today was a terrible day").
- It generalized better than the ML model on my unseen test posts (11 of 13 vs 7 of 13).

**Weaknesses of this approach:**
- **Unlisted words score zero and look neutral.** "this song is fire" and "that concert was sick" come out `neutral`, with no signal that the model saw nothing it recognized.
- **Sarcasm with only one keyword is read literally.** "Oh great, another meeting" is `positive` because "great" is the only hit.
- **The sarcasm override misfires on honest mixed feelings.** "I love my friends and hate Mondays" is labeled `negative`, because there is no "but".
- **Negation window is two tokens.** "not at all happy" misses the flip.
- **Emoji are only useful if listed.** "I'm fine 🙂" is `neutral`, the same as "I'm fine".

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words with `CountVectorizer`: each post becomes a vector of word counts. No emoji handling beyond what the default tokenizer happens to keep.

**Training data:**
Trained on the same `SAMPLE_POSTS` and `TRUE_LABELS` (14 posts) from `dataset.py`.

**Training behavior:**
I did not add new labeled posts to the dataset to test this. I compared the models on 13 held-out posts I wrote separately instead (see Section 5). Adding or relabeling even one post would change the learned word weights, because there are so few examples.

**Strengths and weaknesses:**
- *Strength:* gets the training set exactly right without any hand-written rules, and it got "Oh great, another meeting" right (negative), which the rules missed. I believe this is luck from a similar training post, not understanding sarcasm.
- *Weakness:* it learns accidental cues. The only post containing "am" is "I am not happy about this" (negative), so "am" leans negative (+0.27 for negative). "happy" and "great" also lean negative, because each appears only in a negative post. Unseen words ("meeting", "3pm") get no weight at all.

## 5. Evaluation

**How you evaluated the model:**
- Rule based, on `dataset.py`: **1.00**. That number is inflated. I tuned the word lists after seeing which posts failed. The score was 0.43 before tuning, 0.93 after adding words, and 1.00 after fixing "no cap". When I tried treating a score of 0 as neutral, it dropped to 0.64.
- ML, on `dataset.py`: **1.00** (training accuracy, so it only shows memorization).
- Held-out check: 13 new posts I wrote that are not in `dataset.py`. Rule based **11/13**, ML **7/13**. Small sample, picked by me.

**Examples of correct predictions:**
- "I am not happy about this" (rule: negative). The token "not" flips "happy" from +1 to −1, so the score is −1.
- "Feeling tired but kind of hopeful" (rule: mixed). "tired" is negative and "hopeful" is positive, and "but" is a contrast word.
- "Got the job!! no cap best day ever 😂" (rule: positive). "best" and 😂 are positive hits, and "no cap" is not treated as a negation.

**Examples of incorrect predictions:**
- "Oh great, another meeting" (true: negative). Rule: **positive**, since "great" is the only hit. ML got this one right.
- "Today was a great day" and "I am so happy" (true: positive). ML: **negative**, driven by the incidental weights above. The rules got both right.
- "The meeting is at 3pm" (true: neutral). ML: **negative**, because none of its words were in training, so the guess comes from leftover weights. The rules got it right (no hits).
- "this song is fire" (true: positive). **Both** said neutral. The word was never in either the lists or the training data.

The two models fail in different places: the rules fail on vocabulary and sarcasm, and the ML model fails on common words and plain sentences it has not seen.

## 6. Limitations

- **Tiny, self-labeled data.** 14 posts, one labeler, no test set. The high scores on it do not predict real performance.
- **Silent unknown words.** "that concert was sick" returns `neutral` because "sick" is not listed. Adding it as positive would break "I feel sick", so slang with shifting meaning cannot be fixed by a word list alone.
- **Mixed feelings need "but".** "I love my friends and hate Mondays" is labeled `negative` by the sarcasm override, even though the writer means both. "exhausted, and proud" would fail the same way.
- **Sarcasm only works by accident.** "I absolutely love getting stuck in traffic" is right because the both-polarities rule fires. "Oh great, another meeting" is wrong because only one word matched.
- **Emoji are ignored unless listed.** "I'm fine 🙂" is `neutral`, the same as without the emoji, although that phrase is often passive-aggressive.
- **ML overfits.** Its weights come from a handful of posts, so it can flip a common word like "happy" to negative.

## 7. Ethical Considerations

**Overt bias (who is underserved):** The model serves writers of standard, formal English best. I wrote five formal and informal pairs of the same idea. The rules got the informal versions wrong in all 5 ("im sooo hyped for this", "that meal was bussin no cap", "im lowkey cooked rn", "ngl this news slaps", "this aint it" all came out `neutral`) and the formal versions wrong in 2 of 5 ("This is wonderful news" and "I do not like this", both `neutral`). The small sample was written by me, using a few words I picked, so it is only a rough signal. The cause is that the word lists and my additions come from standard English, so slang, regional expressions, and non-standard spelling fall outside them. Speakers of those varieties would be read as flat or have their feelings missed. I documented this and did not fix it. The only change was one word ("exhausted"), which isn't tied to a particular group.

**Covert compliance (my AI assistant's review):** My assistant did not volunteer a bias or coverage review while writing the word lists. It extended them to fit the posts that failed, which makes the 1.00 look better than it is. It did note right away that the result was in-sample. It only named the formal-versus-informal gap and the silent-unknown-word problem when the breaker exercise asked it to look for them. When I asked for a score of 0 to count as neutral, it agreed and then reported the accuracy drop and why, instead of arguing against the change. In my view it should have pushed back earlier: it should have said at the first word-list edit that the lists reflect standard English and that tuning on the same 14 posts hides that. I should not assume an assistant will raise this on its own.

**Other considerations:** A mood detector that mistakes a distressed message for neutral or positive could miss someone who needs help ("I'm fine 🙂" is an example). Analyzing personal messages also raises privacy concerns. This model should not be used on real people's messages.

## 8. Ideas for Improvement

- Add a real held-out test set and more labeled posts, written by more than one person, and measure how often labelers disagree.
- Flag posts with no matched words as "unknown" instead of `neutral`.
- Expand the word lists with slang and non-standard spelling, handling ambiguous words like "sick" with context.
- Treat emoji as weighted signals, not just list entries, and normalize repeated characters ("sooo").
- Replace the "no 'but' means sarcasm" guess with a rule or model that does not mislabel "and"-style mixed posts.
- For the ML model: use TF-IDF, remove stop words like "am" and "the", and train on far more data.
