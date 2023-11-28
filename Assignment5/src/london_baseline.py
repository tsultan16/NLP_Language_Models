# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from collections import Counter
import pprint

pp = pprint.PrettyPrinter()


# read devset name data from file
correct = 0
total = 0
places = []
for line in open('birth_dev.tsv', encoding='utf-8'):
    place = line.split('\t')[1].strip("\n")
    places.append(place)
    if place == 'London':
        correct += 1
    total += 1                

print(f"Devset London-Baseline, Num Correct: {correct}, Total: {total}, Accuracy: {correct/total}")

counter = Counter(places)
pp.pprint(counter.most_common())
