import sys
maxn = int(sys.argv[2])
with open(sys.argv[1]) as file:
	line = file.readline()
	cnt = 0
	while line and cnt < maxn:
		print(line)
		cnt += 1
		line = file.readline()