---
title: python数据结构基础
date: 2023-04-05 20:49:50
categories: 
 - 编程学习-Python
---

2023-04-03日上传，2023-04-05日第三次更新。
报了个蓝桥杯的python赛道，却没怎么复习……稍微整理了一些数据结构（python实现）的知识点，赛前看一下。
<!-- more -->

**可以先看看第六章python基础**（我记性太差了）

以下有的代码是书上抄的，有的代码是我自己写的，有的代码是gpt生成的，均为经过编译器编译，准确性未知，主要看个思路！

以及：如果网页版看的不爽，您可以点击下载[markdown]()（只是里面没有动图）

### 1. 基础数据结构
#### 1.1 栈
python栈实现：
```python
class Stack:
	def __init__self(self):
		self.items=[]
	def isEmpty(self):
		return self.items==[]
	#入栈
	def push(self,item):
		self.items.append(item)
	#`pop()` 函数用于删除列表中指定索引位置（默认为最后一个元素）的元素
	#出栈
	def pop(self):
		return self.items.pop();
	#取值
	def peek(self):
		return self.items[len(items)-1]
	def size(self):
		return len(self.items)
```
用途：匹配括号，将十进制转化为二进制，前序、中序、后序表达式以及其之间的转换。

#### 1.2 a.队列
python队列的实现
```python
class Queue:
	def __init__(self):
		self.items=[]
	def isEmpty(self):
		return self.items==[]
	#入队
	def enqueue(self,item):
		return self.items.insert(0,item)
	#出队
	def dequeue(self):
		return self.items.pop();
	def size(self):
		return len(self.items)
```
队列的例子：模拟传土豆，模拟打印任务

#### 1.2 b.双端队列
双端队列是与队列类似的有序集合。与队列不同的是，双端队列在哪一端添加元素都没有任何限制，移除也是同理。
双端队列的实现：
```python
class Deque:
	def __init__(self):
		self.items=[]
	def isEmpty(self):
		return self.items==[]
	def addFront(self,item):
		self.items.append(item)
	def addRear(self,item):
		self.items.insert(0,item)
	def removeFront(self):
		return self.items.pop()
	def removeRear(self):
		return self.items.pop(0)
	def size(self):
		return len(self.items)
```
双端队列的应用：回文检测器
回文检测器：将字符串逐字符加入双端队列，依次进行前后端出队，当前端出队元素和后端出队元素相等时再继续。

####  1.3 列表
#####  1.3.1无序列表：链表
node节点是构建链表的基本数据结构。每一个节点至少保有：数据变量；指向下一个节点的应用。下面是一个node类：
```python
class Node:
	def __init__(self,initdata):
		self.data=initdata
		self.next=None
	def getData(self):
		return self.data
	def getNext(self):
		return self.next
	def setData(self,newdata):
		self.data=newdata
	def setNext(self,newnext):
		self.next=newnext
```

通过节点，可以构建无序列表类UnorderedList类
```python
class UnorderedList:
	#在初始化时，只需要一个head节点：
	def __init__(self):
		self.head=None
	def isEmpty(self):
		return self.head==None
	#在head前添加一个节点，并将这个节点设置为head，此时head不再为None，即链表不再为空
	def add(self,item):
		temp=Node(item)
		temp.setNext(self.head)
		self.head=temp
	#通过遍历获取链表长度
	def length(self):
		current=self.head
		cont=0
		while(current!=None):
			cont+=1
			current=current.getNext()
		return cont
	#遍历寻找元素是否在链表内
	def search(self,item):
		current=self.head
		found=false
		while(current!=None and found==false):
			if(current.data==item):
				found=true
				return found
			else:	
				current=current.getNext()
		return found
	#通过遍历删除值
	def remove(self,item):
		current=self.head
		previous=None
		found=False
		while not found:
			if current.getData()==item:
				found=True
			else:
				previous=current
				current=current.getNext()
				
		if previous==None:
			self.head=current.getNext()
		else:
			previous.setNext(current.getNext())
```

#####  1.3.2 有序列表
有序列表中，元素的相对位置取决于他们的基本特征
有序列表的实现：
```python
class OrderedList:
	def __init__(self):
		self.head=None
	#遍历，找到比插入值大的节点，插入到该节点的前面
	def add(self,item):
		current=self.head
		precious=None
		stop=False
		while current!=None and not stop:
			if current.getData()>item:
				stop=True
			else:
				previous=current
				current=current.getNext()
		temp=Node(item)
		if(previous=None):
			self.head=temp
		else:
			temp.setNext(current)
			precious.setNext(temp)
	#遍历，直到满足：1.找到了；2.遍历到的值已经大于要找的值，退出；
	def search(self,item):
		current=self.head
		found=False
		stop=False
		while current!=None and not found and not stop:
			if current.getData()==item:
				found=True
			else:
				if current.getData()>item:
					stop=True
				else:
					current=current.getNext()
		return found
```

### 2. 递归
####  2.1 迷宫搜索
示例：迷宫搜索函数：
下列的实例代码中，接受三个参数：迷宫对象、起始行、起始列
PART_OF_PATH:
```python
def searchForm(maze,startRow,startColumn):
	maze.updatePosition(startRow,startColumn)
	#检查基本情况
	#1.遇到墙
	if (maze[startRow][startColumn]==OBSTACLE):
		return False
	#2.遇到已经走过的格子
	if (maze[startRow][startColumn]==TRIED):
		return False
	#3.找到出口
	if maze.isExit([startRow][startColumn]):
		maze.updatePosition(startRow,startColumn,PART_OF_PATH)
		return True

	maze.updatePosition(startRow,startColumn,TRIED)

	#否则，依次尝试四个方向走动，对于or，只要有一个正确，后面的就不会执行
	found=searchFrom(maze,startRow-1,startColumn)\
		or searchFrom(maze,startRow+1,startColumn)\
		or searchFrom(maze,startRow,startColumn-1)\
		or searchFrom(maze,startRow,startColumn+1)
		
	if found:
		maze.uodatePosition(startRow,startColumn,PART_OF_PATH)
	else:
		maze.uodatePosition(startRow,startColumn,DEAD_END)
	return found
```

####  2.2 汉诺塔
其他示例：汉诺塔
汉诺塔是一种经典的递归问题，它的原理比较简单：有三根杆子，在其中一根杆子上按照大小顺序放置了若干个圆盘，现在需要把这些圆盘从一根杆子移动到另一根杆子上，移动过程中要保证较大的圆盘必须放在较小的圆盘下面，且每次只能移动一个圆盘。在汉诺塔问题中，我们通常称这三根杆子为 A、B、C 杆。

解决汉诺塔问题的常用方法是递归算法，具体的步骤如下：

1.  递归出口：当只有一个圆盘时，直接把它从 A 杆移到 C 杆上，即 return 1。
2.  将 n-1 个圆盘从 A 杆移动到 B 杆上，使用 C 杆作为辅助杆。
3.  将第 n 个圆盘从 A 杆移动到 C 杆上。
4.  将 n-1 个圆盘从 B 杆移动到 C 杆上，使用 A 杆作为辅助杆。

```python
def hanoi(n, A, B, C):
    if n == 1:
        print("Move disk 1 from {} to {}".format(A, C))
        return 1
    else:
        step1 = hanoi(n - 1, A, C, B)
        print("Move disk {} from {} to {}".format(n, A, C))
        step2 = hanoi(n - 1, B, A, C)
        return step1 + 1 + step2
```

####  2.3 动态规划
示例：动态规划找零问题：
+ 问题提出：硬币只有1美分，5美分，10美分和25美分，如何在消耗硬币最少的情况下，找零出37美分？
+  问题剖析：
	 1.递归方法，即：
	 若是每次找出x元的硬币，则剩余找零为（找零-x）元，每次利用函数numCoins寻找四种numCoins（找零-x）中的最小值，构建一颗子节点为4的递归树。

```python
numCoins=min(1+numCoins(找零-1),1+numCoins(找零-5),1+numCoins(找零-10),1+numCoins(找零-25))
```
2. 动态规划方法，即：
每个金额的找零方式都由该金额-i的找零方式所决定，其中，i为硬币面值。
譬如，当我们需要知道15分怎么找时，可以求以下方式的最小值：
+ 一枚一分的硬币+14分所需的最少的硬币（1+6）
+ 一枚5分的硬币+10分所需的最少的硬币（1+1）
+ 一枚10分的硬币+5分所需最少的硬币（1+1）

因此，我们从0开始，指导要解的找零值，求出这之间所有值的最小硬币构成。
下面为动态规划的实现，其中，coinValueList为硬币的面值列表，change表示找零金额，minCoins表示从0到change的所有最优解，coinsUsed存储用于找零的硬币。
```python
def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
	#从0遍历到change（python左闭右开！）
	for cents in range(change+1):
		#所用的硬币数
		coinCount=cents
		#记录使用的硬币的面额
		newCoin=1
		#遍历所有小于找零值的硬币
		for j in [c for c in coinValueList of c<cents]:
			#如果cents-面额j的最小硬币数+1小于之前求得的硬币数，则替换
			if minCoins[cents-j]+1<coinCount:
				CoinCount=minCoins[cents-j]+1
				newCoin=j
		#记载入最优解列表中
		minCoins[cents]=contCount
		#记载入使用过的硬币列表中
		coinUsed[cents]=newCoin
	return minCoins[change]

def printCoins(coinsUsed,change):
	coin=change
	while coin>0:
		thisCoin=coinsUsed[coin]
		print(thisCoin)
		coin=coin-thisCoin		
```

### 3. 搜索与排序
####  3.1 搜索
Python提供了in，通过它可以方便的检查元素是否在列表中：
``15 in [1,2,3,4,15]``

#####  3.1.1 顺序搜索
这个就不讲了，大猩猩都会

#####  3.1.2 二分搜索
目标值比中间元素小，则在左半部分继续查找；否则，在右半部分继续查找。重复以上过程，直到找到目标值或者确定目标值不存在为止。
二分搜索算法的时间复杂度为 O(log n)，其中 n 表示数组中元素的个数，因此它比线性搜索算法的时间复杂度 O(n) 更快，特别是对于大规模的数据集。
显然，二分搜索可以使用递归实现：
```python
def binary_search_helper(arr, target, left, right): 
	if left > right: 
		return -1 
	mid = (left + right) // 2 
	if arr[mid] == target: 
		return mid 
	elif arr[mid] < target: 
		return binary_search_helper(arr, target, mid + 1, right) 
	else: 
		return binary_search_helper(arr, target, left, mid - 1)
```

#####  3.1.3 散列
（这里我不想写了，下面都是gpt写的）
散列（Hashing）是一种用于快速查找的数据结构，它能够在 O(1) 的时间复杂度内完成查找操作。散列使用散列函数将键映射到存储位置，每个存储位置称为散列表中的一个桶。当需要查找一个键时，可以使用散列函数计算出该键对应的桶的位置，并在该桶中查找是否存在对应的值。

散列函数是散列表的关键，它将键映射到桶的位置。一个好的散列函数应该具有以下特点：

1.  散列函数应该能够将不同的键映射到不同的桶，避免不同键的哈希冲突。
2.  散列函数应该尽可能地将键均匀地分布在桶中，避免出现某些桶过载的情况。
3.  散列函数的计算速度应该足够快，否则会影响散列表的性能。

散列函数可以使用不同的算法来实现，包括简单的取余法、乘法散列法、多项式散列法、MD5 散列等。

在实际应用中，散列可以用于解决大量数据的查找问题，例如在数据库中查找记录、在哈希表中查找键值对等。但是，散列也有一些缺点，例如：

1.  散列函数可能出现哈希冲突，即不同的键映射到了同一个桶中，需要使用冲突解决策略来解决。
2.  散列表的性能取决于散列函数的质量和散列表的装载因子，当装载因子过高时，会影响散列表的性能。
3.  散列表的大小通常是固定的，不能动态扩展，如果散列表的大小不够，需要重新创建一个更大的散列表来解决。

下面是一个示例：
```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def hash_func(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self.hash_func(key)
        self.table[index].append((key, value))
    
    def search(self, key):
        index = self.hash_func(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        raise KeyError(key)
    
    def delete(self, key):
        index = self.hash_func(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return
        raise KeyError(key)
```
以上是一个简单的散列表实现，使用 Python 列表作为散列表的存储结构，使用哈希函数将键映射到散列表中的索引位置。当插入一个键值对时，使用哈希函数计算键的索引位置，并将键值对添加到该位置对应的列表中。当查找或删除一个键值对时，同样使用哈希函数计算键的索引位置，并在该位置对应的列表中查找或删除键值对。

这里的哈希函数使用内置函数 hash，它可以将任意 Python 对象转换为整数，并且保证相同的对象具有相同的哈希值。但是，在实际应用中，可以根据具体的应用场景选择不同的哈希函数。

####  3.2 排序
python提供了内置的排序函数：
`list.sort(*, key=None, reverse=False)`：该函数用于对列表进行原地排序，它接受 key 函数和 reverse 参数，如果 reverse 参数为 True，则按照降序排序。

#####  3.2.1 冒泡排序
冒泡排序（Bubble Sort）是一种简单的排序算法，它重复地走访过要排序的数列，依次比较相邻的两个元素，如果它们的顺序错误就交换它们的位置，直到没有需要交换的元素为止。
这没啥要讲的了
{% asset_img 冒泡.gif sucessful %}

#####  3.2.2 选择排序
选择排序（Selection Sort）是一种简单的排序算法，它的基本思想是找到最小元素并将其放置在数组的起始位置，然后继续找到剩余元素中的最小元素并放置在已排序序列的末尾，以此类推，直到所有元素都排好序为止。

{% asset_img 选择.gif sucessful %}

具体实现过程如下：
1.  遍历整个数组，找到其中最小的元素，并记录其位置。
2.  将最小元素与数组的第一个元素进行交换。
3.  排除已排序的第一个元素，对剩余元素执行步骤 1 和 2，直到所有元素都被排序。

下面是一个代码示例：
```python
def selection_sort(lst):
    n = len(lst)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if lst[j] < lst[min_idx]:
                min_idx = j
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst
```

#####  3.2.3 插入排序
一个一个插入到已排序序列中的合适位置，最终完成排序。具体实现过程如下：

1.  遍历整个数组，将数组中的第一个元素视为已排序序列。
2.  遍历未排序序列中的元素，将它插入到已排序序列中的合适位置，使得插入后的序列仍然有序。
3.  重复步骤 2 直到所有元素都被插入到已排序序列中。

{% asset_img 插入.gif sucessful %}

下面是一个代码示例：
```python
def insertion_sort(lst):
    n = len(lst)
    for i in range(1, n):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst
```

#####  3.2.4 希尔排序
希尔排序（Shell Sort）是一种改进的插入排序算法，它是通过将整个序列分成若干个子序列来实现排序，每个子序列分别进行插入排序，最终完成整个序列的排序。

希尔排序的基本思想是将待排序序列按照一定的步长进行分组，对每组使用插入排序算法进行排序。然后将步长逐渐缩小，重复进行分组和排序，直到步长为 1。此时，序列已经被分成了若干个有序子序列，最后进行一次插入排序即可完成整个序列的排序。

{% asset_img 希尔.gif sucessful %}

下面是实现的代码：
```python
def shell_sort(lst):
    n = len(lst)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            key = lst[i]
            j = i - gap
            while j >= 0 and lst[j] > key:
                lst[j + gap] = lst[j]
                j -= gap
            lst[j + gap] = key
        gap //= 2
    return lst
```
其中，参数 lst 是一个待排序的列表，函数返回一个新的已排序的列表。函数首先将整个序列分成若干个子序列，对每个子序列使用插入排序算法进行排序。然后将步长 gap 逐渐缩小，重复进行分组和排序，直到步长为 1。在每个子序列中，将当前元素存储为关键字 key，并将它与已排序序列中的元素进行比较，找到合适的位置并插入，直到所有元素都被插入到已排序序列中为止。

#####  3.2.5 归并排序
归并排序（Merge Sort）是一种稳定的排序算法，它采用分治思想将待排序序列分成若干个子序列，每个子序列都是有序的，然后再将这些有序的子序列合并成一个有序序列。

归并排序的基本思想是将待排序序列不断地对半分割，直到每个子序列只有一个元素，然后将相邻的子序列进行合并，形成新的有序子序列，直到最终只剩下一个有序序列为止。合并操作时，需要额外的一个数组来存储已经排序好的元素，最后再将排序好的元素复制回原数组。

{% asset_img 归并.gif sucessful %}

以下是使用 Python 实现的归并排序代码：
```python
def merge_sort(lst):
    if len(lst) > 1:
        mid = len(lst) // 2
        left_half = lst[:mid]
        right_half = lst[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                lst[k] = left_half[i]
                i += 1
            else:
                lst[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            lst[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            lst[k] = right_half[j]
            j += 1
            k += 1

    return lst
```
其中，参数 lst 是一个待排序的列表，函数返回一个新的已排序的列表。函数首先将待排序序列不断地对半分割，直到每个子序列只有一个元素，然后将相邻的子序列进行合并，形成新的有序子序列，直到最终只剩下一个有序序列为止。在合并操作中，需要额外的一个数组来存储已经排序好的元素，最后再将排序好的元素复制回原数组。

#####  3.2.6 快速排序
快速排序（Quick Sort）是一种常见的排序算法，它采用分治思想将待排序序列分成两个子序列，一部分小于基准元素，一部分大于基准元素。然后对这两个子序列分别进行递归排序，最终得到一个有序序列。

快速排序的基本思想是选定一个基准元素，然后通过一趟排序将待排序序列分成两部分，使得左边的子序列都小于基准元素，右边的子序列都大于基准元素，然后分别对左右两部分递归地进行快速排序，最终得到一个有序序列。

{% asset_img 快速.gif sucessful %}

以下是使用 Python 实现的快速排序代码：
```python
def quick_sort(lst, left, right):
    if left < right:
        pivot_index = partition(lst, left, right)
        quick_sort(lst, left, pivot_index - 1)
        quick_sort(lst, pivot_index + 1, right)

def partition(lst, left, right):
    pivot = lst[left]
    i, j = left + 1, right
    while True:
        while i <= j and lst[i] < pivot:
            i += 1
        while i <= j and lst[j] >= pivot:
            j -= 1
        if i <= j:
            lst[i], lst[j] = lst[j], lst[i]
        else:
            break
    lst[left], lst[j] = lst[j], lst[left]
    return j
```
其中，参数 lst 是一个待排序的列表，参数 left 和 right 是列表的左右边界，函数使用递归的方式实现快速排序。在递归的过程中，首先选定一个基准元素，然后通过 partition 函数将待排序序列分成两部分。partition 函数使用双指针的方式将待排序序列分成两部分，左边部分的元素都小于基准元素，右边部分的元素都大于等于基准元素。最后，将基准元素放在分界点上，返回分界点的位置。然后分别对左右两部分递归地进行快速排序，最终得到一个有序序列。

### 4. 树
不多介绍，直接看实现与算法
#### 4.1 树的实现
##### 4.1.1 实现方法1：列表之列表
例如，a为根节点，有子节点b，c，节点b又有子节点d（左），则使用列表：
```python
[a,[b,[d,[],[]],[]],[c,[],[]]]
```
不说了，这方法真的会有人用吗

##### 4.1.2 实现方法2：节点与引用
基础类：
```python
class BinaryTree:
	def __init__(self,rootObj):
		self.key=rootObj
		self.leftChild=None
		self.rightChild=None
```
插入左节点（如果已经存在左子节点，插入时要把原先的左子节点降一层，自己到那个位置）：
```python
def instertLeft(self,newNode):
	if(self.leftChild==None):
		self.leftChild=BinaryTree(newNode)
	else:
		t=BinaryTree(newNode)
		t.leftChild=self.leftChild
		self.leftChild=t
```
右子节点也一样。
访问函数：
```python
def getRightChild(self):
	return self.rightChild
def getLeftChild(self):
	return self.LeftChild
def setRootVal(self,obj):
	self.key=obj
def getRootVal(self):
	return self.key
```

#### 4.2 树的遍历
树的遍历方式分为三种：前序遍历、中序遍历、后序遍历

##### 4.2.1 前序遍历
先访问根节点，然后递归地前序遍历左子树，最后递归地前序遍历右子树
{% asset_img 前序.gif sucessful %}
```python
def preorder(tree):
	if tree:
		print(tree.getRootVal())
		preorder(tree.getLeftChild())
		preorder(tree.getRightChild())
```

##### 4.2.2 中序遍历
先递归地中序遍历左子树，然后访问根节点，在中序遍历右子树
{% asset_img 中序.gif sucessful %}
```python
def inorder(tree):
	if tree:
		inorder(tree.getLeftChild())
		print(tree.getRootVal())
		inorder(tree.getRightChild())
```

##### 4.2.3 后序遍历
先递归地后序遍历右子树，然后递归地后序遍历左子树，最后访问根节点
{% asset_img 后序.gif sucessful %}
```python
def postorder(tree):
	if tree:
		postorder(tree.getLeftChild())
		postorder(tree.getRightChild())
		print(tree.getRootVal())
```

#### 4.3 利用二叉堆实现优先级队列
二叉堆（binary heap）是一种特殊的二叉树数据结构，它可以用数组来实现，并且常用于实现优先队列。

二叉堆分为最大堆和最小堆两种类型。最大堆满足任何一个父节点的键值大于等于它的任何一个子节点的键值；最小堆则满足任何一个父节点的键值小于等于它的任何一个子节点的键值。因此，最大堆的堆顶是堆中的最大元素，最小堆的堆顶是堆中的最小元素。

二叉堆的数组实现中，数组的第一个元素是根节点，数组的下标从1开始（而不是0）。对于第i个节点，它的左子节点在2i的位置，右子节点在2i+1的位置，父节点在i/2的位置（向下取整）。

二叉堆的主要操作包括插入一个元素、删除堆顶元素和查找堆顶元素。插入一个元素需要将元素插入到堆的最后一个位置，然后通过向上逐级比较交换，将新元素移动到合适的位置以维护堆的性质。删除堆顶元素需要将堆顶元素与堆中最后一个元素交换位置，然后删除最后一个元素，并通过向下逐级比较交换，将堆顶元素移动到合适的位置以维护堆的性质。查找堆顶元素只需要返回数组的第一个元素。

##### 4.3.1 二叉堆的实现
在实现二叉堆时，我们通过创建一颗完全二叉树来维持树的平衡。在完全二叉树中，除了最底层，其他每一层的节点都是满的。在对底层，从左往右填充节点。

完全二叉树的特殊之处在于，可以用一个列表来表示它。
完全二叉树节点在列表中的排序为：从上而下，从左往右。
按此排布，对于在列表中位置p的节点来说，其左子节点的位置为2p，右子节点的位置为2p+1（请读者自行推算一遍）

二叉堆的实现：
```python
def __init__(self):
	self.heapList=[0]
	self.currentSize=0
```

对于一个要求子节点大于父节点的堆来说，插入的最简单、最高效的方法就是将元素追加到列表的末尾，然后和其父节点比较，若小于其父节点，就对换位置。此处代码不再给出，简单来说就是比较插入的第i个和父节点第i//2个对比。

二叉堆还有一个功能，删除最小元素（即根节点），但是删除后，需要对堆进行重建。重建方法为：
1. 取出列表中最后一个元素，将其放在根节点的位置
2. 通过与子节点之间的交换，将新的根节点沿着树推到正确的位置

#### 4.4 二叉搜索树
二叉搜索树依赖于这样的性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。

##### 4.4.1 插入
根据性质，插入算法为：
1. 从根节点开始搜索二叉树，比较新键与当前节点的键，如果新键更小，搜索左子树，如果新键更大，搜索右子树
2. 当没有可供搜索的左/右字节点时，说明找到了正确的位置。
3. 创建一个TreeNode对象，并将其插入到前一步发现的位置上。

下面是一个利用递归方法实现的代码：
```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            self._insert(val, self.root)

    def _insert(self, val, node):
        if val < node.val:
            if not node.left:
                node.left = Node(val)
            else:
                self._insert(val, node.left)
        else:
            if not node.right:
                node.right = Node(val)
            else:
                self._insert(val, node.right)
```

##### 4.4.2 查找
相似的，查找的方法为：
```python
def search(self, val):
    node = self.root
    while node:
        if node.val == val:
            return node
        elif val < node.val:
            node = node.left
        else:
            node = node.right
    return None
```

##### 4.4.3 删除
删除的情况就相对复杂。
1.  要删除的节点是叶子节点，也就是没有左右子节点的节点。在这种情况下，可以直接删除该节点，将其父节点的左子节点或右子节点指向 None。
2.  要删除的节点只有一个子节点，可以将该节点的子节点替换为该节点。
3.  要删除的节点有两个子节点。在这种情况下，可以将该节点的左子树的最大节点或右子树的最小节点替换该节点。（因为左子树的最大节点或右子树的最小节点正好可以填进去）

```python
class BST:
    # 省略插入方法和查找方法...
    
    def delete(self, val):
        node, parent = self.search_with_parent(val)
        if not node:
            return False  # 未找到要删除的节点
        if not node.left or not node.right:  # 第一种和第二种情况
            child = node.left or node.right
            if parent:
                if node == parent.left:
                    parent.left = child
                else:
                    parent.right = child
            else:
                self.root = child
        else:  # 第三种情况
            succ = node.right
            while succ.left:
                succ = succ.left
            node.val = succ.val
            if succ == node.right:
                node.right = succ.right
            else:
                p = self.search_with_parent(succ.val)[1]
                p.left = succ.right
        return True
```

#### 4.5 平衡二叉搜索树
这个真太烦了，我不高兴看了T.T

#### 4.6 霍夫曼树
霍夫曼树（Huffman Tree）是一种带权路径最短的树，通常用于数据压缩。它的构建过程基于贪心算法，根据数据频率构建一棵无损压缩的树。具体来说，给定一组数据和对应的权值，霍夫曼树的构建过程如下：

1.  对所有的数据按照权值从小到大排序，每个数据作为一个单独的节点。
2.  每次从排序后的节点中选取权值最小的两个节点，将它们合并为一个新节点，新节点的权值为两个节点的权值之和，左子节点为权值较小的节点，右子节点为权值较大的节点。
3.  将新节点插入到排序后的节点列表中，并删除原来的两个节点。
4.  重复步骤2和3，直到只剩下一个节点为止，该节点即为霍夫曼树的根节点。

{% asset_img 哈夫曼.gif sucessful style="display: block; margin: 0 auto; max-width: 50%;"%}
下面是一个Python实现示例：
```python
class HuffmanNode:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.left = None
        self.right = None

def build_huffman_tree(data):
    # 构建叶子节点列表
    nodes = [HuffmanNode(val, weight) for val, weight in data]
    # 构建霍夫曼树
    while len(nodes) > 1:
        # 按权值排序
        nodes.sort(key=lambda node: node.weight)
        # 取出权值最小的两个节点
        left_node = nodes.pop(0)
        right_node = nodes.pop(0)
        # 构建新节点
        new_weight = left_node.weight + right_node.weight
        new_node = HuffmanNode(None, new_weight)
        new_node.left = left_node
        new_node.right = right_node
        # 将新节点加入节点列表中
        nodes.append(new_node)
    # 返回根节点
    return nodes[0]
```

在这个实现中，我们先定义了一个`HuffmanNode`类，表示霍夫曼树中的节点。每个节点包含一个`value`属性表示节点的值（如果节点是叶子节点，则为原始数据），一个`weight`属性表示节点的权值，以及左子节点和右子节点。我们还定义了一个`build_huffman_tree(data)`函数，用于构建霍夫曼树。`data`参数是一个二元组列表，每个二元组包含一个数据和对应的权值。函数返回霍夫曼树的根节点。

在构建完霍夫曼树后，我们可以通过对树进行遍历来获得每个数据的编码。具体来说，我们可以对树进行先序遍历，在遍历过程中，记录每个叶子节点的编码（0表示向左走，1表示向右走）。最终得到的编码就是霍夫曼编码，可以用于数据压缩。

例如，如果一个数据出现的频率很高，代表着他权值很高，我们可以用一个比较短的编码来表示它，比如说用一个1位的编码表示它。而如果一个数据出现的频率很低，我们可以用一个比较长的编码来表示它，比如说用一个10位的编码表示它。这样，在对数据进行编码后，数据的存储空间就会减少。

### 5. 图
下面先回顾一下图中的术语以及定义：
1.  顶点（vertex）：也称为节点，表示图中的一个点，通常用一个唯一的标识符来标识。
2.  边（edge）：表示两个顶点之间的连线，可以是有向或无向的，可以有权重或无权重。
3.  权重（weight）：如果边带有数值，则称这个数值为边的权重。
4.  路径（path）：表示从一个顶点到另一个顶点依次经过的边和顶点的序列，路径的长度为路径上所有边的权重之和。
5.  环（cycle）：表示一个顶点经过一系列边回到自身的路径。
6.  连通（connected）：如果图中的任意两个顶点都有一条路径相连，则称该图是连通的。
7.  连通分量（connected component）：无向图中每个连通的部分称为一个连通分量。
8.  强连通（strongly connected）：如果有向图中的任意两个顶点都有互相到达的路径，则称该图是强连通的。
9.  强连通分量（strongly connected component）：有向图中每个强连通的部分称为一个强连通分量。
10.  入度（in-degree）：有向图中指向一个顶点的边的数量。
11.  出度（out-degree）：有向图中从一个顶点出发的边的数量。
12.  邻接点（adjacent vertex）：与一个顶点直接相连的顶点称为它的邻接点。
13.  邻接矩阵（adjacency matrix）：用矩阵来表示图中每个顶点之间的连通关系，其中矩阵中的行和列分别代表图中的顶点，矩阵中的元素表示两个顶点之间是否有连通关系。
14.  邻接表（adjacency list）：用链表来表示图中每个顶点的邻接点列表，链表中的每个节点表示一个邻接点。

#### 5.1 图的抽象数据类型以及其实现
##### 5.1.1 邻接矩阵
只是一个矩阵，不多赘述
##### 5.1.2 邻接表
python中，字典的键值对可以有效地实现图的边。下面是一个示例，利用字典connectedTo来记录与节点Vertex相邻的点。

以下给出一个节点类：
```python
class Vertex:
	def __init__(self,key):
		self.id=key
		self.connectedTo={}
	def addNeighbor(self,nbr,weight=0):
		self.connectefTo(nbr)=weight
	def __str__(self):
		return str(self.id)+' connectedTo: '+str([x.id for x in self.connectedTo])
	def getConnections(self):
		#通过获取所有键，获取节点所有相接的顶点名
		return self.connectedTo.keys()
	def getId(self):
		return self.id
	def getWeight(self,nbr):
		return self.connectedTo[nbr]
```

Graph类就不再赘述，只需要特别注意两个功能：添加顶点，添加边
#### 5.2 广度优先搜索（BFS）
要求边的权值都为0！！！！！！

BFS，即广度优先搜索（Breadth-First Search），是一种图的遍历算法，用于在图中搜索特定的节点或路径。BFS从给定的起始节点开始遍历图，首先访问起始节点的所有邻居节点，然后按照遍历的深度依次访问下一层节点，直到遍历完整张图或找到目标节点为止。

BFS通常借助队列（Queue）数据结构来实现。首先将起始节点加入队列中，然后不断从队列中取出最早进入队列的节点，并将其邻居节点加入队列中（如果之前取过了，则不取），并计算节点到起始节点的距离（父节点到起始节点的距离+1）直到队列为空或找到目标节点为止。

{% asset_img 广度优先搜索.gif sucessful style="display: block; margin: 0 auto; max-width: 50%;"%}

下面是一个实现的代码：
```python
from pythonds.graphs import Graph, Vertex
from pythonds.basic import Queue

def BFS(g,start,end):
	#储存据起始节点距离
	distance={}
	distance[start]=0
	#储存是否有被访问过
	visited=[]
	#队列
	queue=Queue()
	queue.enqueue(start)
	#创建字典，用于记录每个节点的父节点，用于回溯路径
	parent = {} 
	while(!queue.size==0):
		#currentq：当前访问到的节点，名字起错了，不想改了
		currentq=queue.dequeue()
		if(currentq==end):
			#回溯打印
			while(!currentq==start):
				print(currentq.id)
				currentq=parent[currentq]
			print(start.id)
			return True
			
		for item in currentq.connectedTo:
			if(item not in visited):
				visited.appenf(item)
				queue.enqueue(item)
				distance[item]=distance[current]+1
				parent[item]=currentq
	return False	
```
#### 5.3 深度优先搜索
深度优先搜索（Depth-First-Search, DFS）是一种用于遍历或搜索树或图的算法，其主要思想是从起点开始，不断往深度方向搜索，直到找到目标节点或者无法继续为止，然后返回上一层节点，继续搜索其他未被访问过的节点。

递归式的DFS可以通过递归调用实现。具体实现过程如下：

1.  创建一个visited数组，用于记录每个节点是否被访问过，初始值都为False。
2.  定义DFS函数，输入参数为当前节点和图的邻接表表示。首先标记当前节点为已访问，并输出当前节点。然后遍历当前节点的邻居节点，对于每个未被访问过的邻居节点，递归调用DFS函数。
3.  在主函数中遍历图中的每个节点，对于每个未被访问过的节点，调用DFS函数。

以下是Python实现代码示例：
```python
def DFS(currentVertex, visited)#当前访问的节点，已经访问过的节点列表
	visited[currentVertex]=True
	print(currentVertex.id)
	for item in currentVertex.connectedTo:
		if(item not in visited):
			DFS(currentVertex,visited)
```

#### 5.4 最短路径
##### 5.4.1 Dijkstra算法
Dijkstra算法是一种用于求解单源最短路径的贪心算法，它能够计算出从一个源点到图中其他所有点的最短路径。算法的基本思想是维护一个到源点的距离数组，每次选取一个距离最短的点进行松弛操作（通过更新边来减小从起点到顶点的距离），更新其他节点到源点的距离值。当所有节点都被更新后，最短路径就求解完成。

以下是使用Python实现Dijkstra算法的代码
```python
# 定义一个函数用于实现Dijkstra算法
def dijkstra(graph, start):
    # 初始化距离字典，所有节点的距离值默认为无限大
    dist = {node: float('inf') for node in graph}
    # 将起点的距离值初始化为0
    dist[start] = 0
    ……#初始化距离字典
    # 初始化路径字典
    path = {start: []}
    # 将起点放入已访问节点集合中
    visited = set()

    while True:
        # 从未访问节点中找到距离起点最近的节点
        min_node = None
        for node in graph:
            if node not in visited:
                if min_node is None:
                    min_node = node
                elif dist[node] < dist[min_node]:
                    min_node = node

        if min_node is None:
            break

        # 将该节点标记为已访问
        visited.add(min_node)

        # 更新所有与该节点相邻的节点的距离值
        for neighbor, weight in graph[min_node].items():
            if neighbor not in visited:
                new_dist = dist[min_node] + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    path[neighbor] = path[min_node] + [min_node]

    return dist, path
```
##### 5.4.2 Prim算法
Prim算法是一种用于解决最小生成树问题的贪心算法。其基本思路是从一个起点开始，不断扩展生成树，每次加入距离已有部分最近的一个点，直到所有点都被加入为止。

具体步骤如下：
1.  初始化一个空的生成树，以一个节点作为起始节点；
2.  找到与生成树相邻的边中，权值最小的那条边，将其连接的点加入生成树中；
3.  重复第二步，直到所有节点都加入生成树中。

该算法可以借助优先级队列来实现（Dijkstra算法也可以，但我没用），优先级队列可以弹出包含最小元素的字典。

下面是一个python实现：
```python
from pythonds.graphs import PriorityQueue, Graph, Vertex
def Prim(G, start):
	pq=PriorityQueue()
	for v in G:
		v.setDistance(sys.maxsize)#设置和起始点的距离
		v.setPred(None)#设置前驱结点
	start.SetDistance(0)
	#创建一个堆，存放河节点的相邻节点
	pq.buildHeap([(v.getDistance, v) for v in G])
	while not pq.isEmpty():
		#取出最小的起始点的相邻节点
		currentVert=pq.delMin()
		#遍历该节点的所有相邻节点
		for nextVert in currentVert.getConnections():
			#新的距离=节点currentVert和原点的距离+当前节点和currentVert的距离
			newCost=currentVert.getWeight(nextVert)+currentVert.getDistance()
			#如果新生成的路径长度比原来的短
			if v in pq and newCost<nextVert.getDistance():
				nextVert.setPred(currentVert)
				nextVert.setDistance(newCost)
				pq.decreaseKey(nextVert,newCost)
```

### 6. python基础复习
#### 6.1 python基础库
1. 队列：queue，入队和出队的方法分别是put()和get()，empty(): 判断队列是否为空。qsize(): 返回队列中当前的元素个数。
2. 栈：Python 中没有专门表示栈的基础库，但是可以使用内置的 list类来实现栈的功能，因为list 的 append() 和 pop()方法可以分别实现入栈和出栈操作。

#### 6.2 基础函数
##### 6.2.1 列表基础函数
1.  append：在列表末尾添加一个元素，例如：`list.append(item)`
2.  extend：将一个列表中的所有元素添加到另一个列表末尾，例如：`list.extend(another_list)`
3.  insert：在列表的指定位置插入一个元素，例如：`list.insert(index, item)`
4.  remove：删除列表中指定的元素，例如：`list.remove(item)`
5.  pop：从列表中删除指定位置的元素，并返回该元素，例如：`list.pop(index)`
6.  index：返回列表中指定元素的索引位置，例如：`list.index(item)`
7.  count：返回列表中指定元素出现的次数，例如：`list.count(item)`
8.  sort：对列表中的元素进行排序，例如：`list.sort()`
9.  reverse：将列表中的元素反向排序，例如：`list.reverse()`
10.  clear：从列表中删除所有元素，例如：`list.clear()`

##### 6.2.2 字典基础函数
1.  dict()：创建一个新字典
2.  len(dict)：返回字典中键值对的数量
3.  dict[key]：获取字典中指定键的值
4.  dict[key] = value：设置字典中指定键的值
5.  del dict[key]：从字典中删除指定键
6.  key in dict：检查字典中是否包含指定键
7.  dict.keys()：返回一个包含字典所有键的列表
8.  dict.values()：返回一个包含字典所有值的列表
9.  dict.items()：返回一个包含字典所有键值对的列表
10.  dict.get(key, default)：获取字典中指定键的值，如果键不存在返回默认值
11.  dict.setdefault(key, default)：获取字典中指定键的值，如果键不存在设置默认值并返回
12.  dict.pop(key, default)：从字典中删除指定键，并返回其对应的值。如果键不存在返回默认值
13.  dict.update(other_dict)：使用其他字典中的键值对来更新当前字典

##### 6.2.3 map、filter函数
`map()` 是 Python 内置的一个高阶函数，它将一个函数作用于一个或多个可迭代对象的每个元素上，并返回一个可迭代对象（迭代器），其中包含应用函数后的结果。

`map()` 函数的语法如下，下面是一个简单的例子，将一个列表中的每个元素乘以 2：：
```python
def double(x):
    return x * 2

numbers = [1, 2, 3, 4, 5]
doubled_numbers = list(map(double, numbers))
print(doubled_numbers)  # [2, 4, 6, 8, 10]
```
在这个例子中，`double()` 函数接收一个参数并返回该参数的两倍。`map()` 函数将 `double()` 应用于 `numbers` 列表中的每个元素，并返回一个迭代器，其中包含应用函数后的结果。`list()` 函数将迭代器转换为一个列表，最终输出 `[2, 4, 6, 8, 10]`。

`filter`函数：对一个可迭代对象中的元素进行过滤，返回满足条件的元素组成的新的可迭代对象。
```python
def is_even(num):
    return num % 2 == 0

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_nums = filter(is_even, nums)
print(list(even_nums))  # [2, 4, 6, 8, 10]
```
##### 6.2.4 字符串函数
1.  len()：获取字符串的长度。
2.  strip()：去掉字符串开头和结尾的空格。
3.  split()：按照指定的分隔符将字符串分割成列表。
4.  join()：将列表或元组中的字符串拼接成一个字符串，中间用指定的字符隔开。
5.  replace()：将字符串中指定的子字符串替换为另一个字符串。
6.  find()：查找指定的子字符串在字符串中的位置，返回第一个匹配到的位置。
7.  lower()和upper()：分别将字符串转换成小写和大写。
8.  startswith()和endswith()：判断字符串是否以指定的字符串开头或结尾。
9.  isdigit()、isalpha()和isalnum()：判断字符串是否全部由数字、字母或数字字母组成。
10.  format()：将指定的值格式化到字符串中。
11. list()：转化为一个列表

示例：join的用法：
```python
words = ['Python', 'is', 'awesome']
sep = ' '
sentence = sep.join(words)
print(sentence) # 输出：Python is awesome
```
