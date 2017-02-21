import json, sys, os.path, os
def comp(obj, dep):
    '''
    判断目标文件是否比依赖文件新，如果是返回True
    '''
    min1 = -1
    for e in obj:
        if not os.path.exists(e):
            print("%s doesn't exists!" % e)
            sys.exit(1)
        t = os.path.getmtime(e)
        if min1 < 0 or t < min1:
            min1 = t
    max2 = 0
    for e in dep:
        if not os.path.exists(e):
            print("%s doesn't exists!" % e)
            sys.exit(1)
        t = os.path.getmtime(e)
        if t > max2:
            max2 = t
    return min1 > max2
#
def check(node, path, data, deps):
    '''自上而下检查依赖关系树，如果子节点出现过时的状态，依赖它的所有父节点都要更新'''
    if node in path:
        print('Circular Dependency!')
        sys.exit(1)
        
    #检查该节点的依赖的所有节点
    status = False
    path.add(node)
    for k in range(len(deps)):
        if deps[node][k]:
            status = check(k, path, data, deps) or status
    path.remove(node)
    #子节点过或者当前节点的状态过时都要执行更新命令
    if status or (not comp(data[node]['obj'], data[node]['dep'])):
        for e in data[node]['act']:
            print(e)
            os.system(e)
        status = True
    return status
#
if len(sys.argv) < 2:
	fileName = 'Makefile.json'
else:
	fileName = sys.argv[1]
data = json.load(open(fileName))
N = len(data)
#依赖关系用一个二维数组表示，deps[n][k]如果为True，那么n节点就依赖k节点
deps = [[] for k in range(N)]
for k in range(N):
	deps[k] = [False for k in range(N)]
for m in range(N - 1):
	for n in range(m+1, N):
     #自动判断依赖关系，如果节点间的obj和dep由交集，那么两者就存在依赖关系
		if len(set(data[m]['obj']) & set(data[n]['dep'])) > 0:
			deps[n][m] = True
		if len(set(data[n]['obj']) & set(data[m]['dep'])) > 0:
			deps[m][n] = True
		if deps[m][n] and deps[n][m]:
			print('Circular Dependency!')
#找到根节点，依次check
for k in range(N):
    f = True
    for n in range(N):
        if deps[n][k]:
            f = False
            break
    if f:
        check(k, set(), data, deps)
#