#!/usr/bin/env python3

import sys
import pyconll
import itertools as it
from collections import defaultdict

in_file_path = sys.argv[1]
corpus = pyconll.iter_from_file(sys.argv[1])

word_to_tag = defaultdict(set)
tag_to_word = defaultdict(set)
tag_to_tag = defaultdict(set)

for sentence in corpus:
    for i in range(len(sentence)):
        word = sentence[i].form
        tag = sentence[i].upos
        word_to_tag[word].add(tag)
        tag_to_word[tag].add(word)
        if i < len(sentence) - 1:
            next_tag = sentence[i+1].upos
            tag_to_tag[tag].add(next_tag)

word_to_tag_counts = defaultdict(lambda: 0)
tag_to_word_counts = defaultdict(lambda: 0)
tag_to_tag_counts = defaultdict(lambda: 0)

for group in word_to_tag.values():
    word_to_tag_counts[len(group)] += 1

for group in tag_to_word.values():
    tag_to_word_counts[len(group)] += 1

for group in tag_to_tag.values():
    tag_to_tag_counts[len(group)] += 1

print("word -> tag")
for k in sorted(word_to_tag_counts.keys()):
    print(word_to_tag_counts[k], "words with", k, "tags")
print()

print("tag -> word")
for k in sorted(tag_to_word_counts.keys()):
    print(tag_to_word_counts[k], "tags with", k, "words")
print()

print("tag -> tag")
for k in sorted(tag_to_tag_counts.keys()):
    print(k, "tags with", tag_to_tag_counts[k], "successors")
print()

print(len(word_to_tag.keys()), "words")
print(len(tag_to_word.keys()), "tags")
