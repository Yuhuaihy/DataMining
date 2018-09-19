import sys
import os
import collections
import re
from IPython import embed
# Author: Yu Huai
# Homework0

def Apriori(path):
	totalWords = []
	l1 = ''
	l2 = ''
	l2 = ''
	files = os.listdir(path)
	n = len(files)
	reg = r'\W'
	wordList = []

	for i in range(n):
		filepath = os.path.join(path, files[i])
		with open(filepath, 'r') as f:
			data = f.readlines()
			filewords = set()
			for line in data:
				words = re.split(reg, line)
				temp = set(words)
				temp.remove('')
				filewords.update(temp)
			totalWords.append(filewords)
			wordList += list(totalWords[i])
	
	itemSet1 = dict(collections.Counter(wordList))
	itemSet1 = sorted(itemSet1.items(), key=lambda itemSet1:itemSet1[1], reverse=True)
	l1 = itemSet1[0][0]
	wordNum = len(itemSet1)
	itemSet2 = []
	miniSupport = itemSet1[0][1] + itemSet1[1][1] - n 
	
	for i in range(wordNum-1):
		if itemSet1[i][1] < miniSupport:
			break
		ele1 = itemSet1[i][0]
		for j in range(i+1, wordNum):
			if itemSet1[j][1] < miniSupport:
				break
			ele2 = itemSet1[j][0]
			itemSet2.append([ele1,ele2,0])

	for ele in itemSet2:
		for words in totalWords:
			if ele[0] in words and ele[1] in words:
				ele[2] += 1
	
	itemSet2 = sorted(itemSet2, key = lambda itemSet2:itemSet2[2], reverse=True)
	l2 = [itemSet2[0][0], itemSet2[0][1]]
	#embed()
	itemSet3 = []
	numSet2 = len(itemSet2)
	for i in range(numSet2-1):
		temp = set()
		for j in range(i+1, numSet2):
			temp = set([itemSet2[i][0], itemSet2[i][1], itemSet2[j][0], itemSet2[j][1]])
			if len(temp) == 3:
				itemSet3.append(list(temp)+[0])
	
	for ele in itemSet3:
		for words in totalWords:
			if ele[0] in words and ele[1] in words and ele[2] in words:
				ele[3] += 1
	
	maxFreq = 0
	for item in itemSet3:
		if item[3] > maxFreq:
			l3 = [itemSet3[0][0], itemSet3[0][1], itemSet3[0][2]]
			maxFreq = item[3]

	return l1,l2,l3


if __name__ == '__main__':
	filepath = sys.argv[1]
	print(filepath)
	result1, result2, result3 = Apriori(filepath)
	print ('word1:  {}'.format(result1))
	print ('word1, word2: {} : '.format(result2))
	print ('word1, word2, word3: {}'.format(result3))



			




