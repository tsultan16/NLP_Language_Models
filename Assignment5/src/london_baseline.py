# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

# read devset name data from file
correct = 0
total = 0
for line in open('birth_dev.tsv', encoding='utf-8'):
    place = line.split('\t')[1].strip("\n")
    if place == 'London':
        correct += 1
    total += 1                
            
print(f"Devset London-Baseline, Num Correct: {correct}, Total: {total}, Accuracy: {correct/total}")