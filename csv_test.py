#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.7.24
# Version     :   1.0

import csv

with open('test.csv','wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow([1,211.22])

with open('test.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	print(type(reader.next()))
	for row in reader:
		print ','.join(row)
