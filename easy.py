







# 1304. Find N Unique Integers Sum up to Zero
class Solution:
    def sumZero(self, n: int) -> List[int]:
        a = n//2
        ans=[i for i in range(-a,a+1)]
        if not n&1:
            ans.pop(a)
        return ans


# 1252. Cells with Odd Values in a Matrix
class Solution:
    def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
        x = [0]*n
        y = [0]*m
        
        for i in indices:
            x[i[0]]+=1
            y[i[1]]+=1
        ans = 0
        for i in x:
            for j in y:
                ans+=(1 if (i+j)&1 else 0)
        return ans


# 1266. Minimum Time Visiting All Points
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        ans = 0
        p = points[0]
        for q in points[1:]:
            ans+=max(abs(q[0]-p[0]),abs(q[1]-p[1]))
            p = q
        return ans


# 1295. Find Numbers with Even Number of Digits
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        ans = 0
        for i in nums:
            t = 0
            while i:
                t+=1
                i//=10
            ans+=(0 if t&1 else 1)
        return ans


# 88. Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        while n >= 1:
            i = n+m-1
            if m <= 0 or nums2[n-1] >= nums1[m-1]:
                nums1[i] = nums2[n-1]
                n -= 1
            else:
                nums1[i] = nums1[m-1]
                m -= 1



# 66. Plus One
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        t = 1
        ans = []
        for i in digits[-1::-1]:
            ans = [(i+t)%10] + ans
            t = (i+t)//10
        if t!= 0:
            ans = [t]+ans
        return ans


# 1313. Decompress Run-Length Encoded List
class Solution:
    def decompressRLElist(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        for i in range(0,n-1,2):
            ans+=[ nums[i+1] for j in range(nums[i]) ]
        return ans


# 405. Convert a Number to Hexadecimal
class Solution:
    def toHex(self, num: int) -> str:
        if num == 0:
            return '0'
        if num <0:
            num = num&0xffffffff
        ans = ''
        while num:
            a = num&15
            num >>= 4
            ans = (str(a) if a<10 else chr(a+87)) + ans
        return ans


# 190. Reverse Bits
class Solution:
    def reverseBits(self, n: int) -> int:
        ans = 0
        for i in range(32):
            if n&(1<<i):
                ans += 1<< 31-i
        return ans


# 342. Power of Four
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        if num < 0 :
            return False
        a = 0
        i = 0
        while num:
            if i&1 and num&1:
                return False
            a += num&1
            num >>= 1
            i += 1
        return (True if a==1 else False)


# 231. Power of Two
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n < 0:
            return False
        a = 0
        while n:
            a += (n&1)
            n >>= 1
        return (True if a==1 else False)



# 693. Binary Number with Alternating Bits
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        t = n&1
        n >>= 1
        while n:
            if t == n&1:
                return False
            t = n&1
            n >>= 1
        return True


# 268. Missing Number
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        a = [i for i in range(len(nums)+1)]
        ans = sum(a)-sum(nums)
        return ans


# 371. Sum of Two Integers
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xffffffff
        
        # works both as while loop and single value check 
        while (b & mask) > 0:
            
            carry = ( a & b ) << 1
            a = (a ^ b) 
            b = carry
        
        # handles overflow
        return (a & mask) if b > 0 else a


# 371. Sum of Two Integers
var getSum = function(a, b) {
    c = a^b;
    d = a&b;
    while(d){
        t = c;
        c = c^(d<<1);
        d = t&(d<<1);
    } 
    return c;
};


# 169. Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        c = 0
        t = 0
        for n in nums:
            if c==0:
                t = n
            c += (1 if n==t else -1)
        return t



# 389. Find the Difference
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        S = list(s) + list(t)
        ans = 0
        for i in S:
            ans ^= ord(i)
        return chr(ans)



# 762. Prime Number of Set Bits in Binary Representation
class Solution:
    def countPrimeSetBits(self, L: int, R: int) -> int:
        def isP(n):
            if n <= 1:
                return False
            if n == 2:
                return True
            for k in range(2,n):
                if n%k == 0:
                    return False
            return True
        ans = 0
        for i in range(L,R+1):
            p = 0
            for j in range(32):
                if i&(1<<j):
                    p+=1
            if isP(p):
                ans+=1  
        return ans


# 136. Single Number
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            ans ^= n
        return ans


# 476. Number Complement
class Solution:
    def findComplement(self, num: int) -> int:
        i = 0
        b = num
        while b:
            num ^= (1<<i)
            b >>= 1
            i += 1
        return num


# 461. Hamming Distance
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        a = x^y
        ans = 0
        for i in range(32):
            if a&(1<<i):
                ans += 1 
        return ans


# 461. Hamming Distance
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        a = x^y
        ans = 0
        while a:
            if a&1:
                ans+=1
            a = a>>1
        return ans



# 1290. Convert Binary Number in a Linked List to Integer
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        ans=0
        while head:
            ans = (ans<<1)+head.val
            head = head.next
        return ans


# 191. Number of 1 Bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while(n):
            ans += (n&1)
            n = n>>1
        return ans


# 191. Number of 1 Bits
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        for i in range(32):
            if n&(1<<i):
                ans+=1
        return ans


# 1277. Count Square Submatrices with All Ones
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0]*(n+1) for i in range(m+1)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    dp[i+1][j+1] = min(dp[i][j],dp[i+1][j],dp[i][j+1]) + 1
                    
        ans = sum(map(sum,dp))
        return ans
            
            


# 1314. Matrix Block Sum
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        if not mat:
            return 0
        m = len(mat)
        n = len(mat[0])
        
        dp = [[0]*n for i in range(m)]
        re = [[0]*n for i in range(m)]
        
        for i in range(m):
            s = 0
            for j in range(n):
                s+=mat[i][j]
                dp[i][j] = s
                if i > 0:
                    dp[i][j]+=dp[i-1][j]

        for i in range(m):
            for j in range(n):
                a, b = min(i+K,m-1),min(j+K,n-1)
                a_,b_ = max(i-K,0),max(j-K,0)

                re[i][j] = dp[a][b]
                if a_ > 0:
                    re[i][j] -= dp[a_-1][b]
                if b_ > 0:
                    re[i][j] -= dp[a][b_-1]
                if a_ > 0 and b_ > 0:
                    re[i][j] += dp[a_-1][b_-1]
                
        return re
            
# 303. Range Sum Query - Immutable
class NumArray:

    def __init__(self, nums: List[int]):
        if not nums:
            return
        self.f = [0] * (len(nums)+1)
        self.f[0] = nums[0]
        for i in range(1,len(nums)):
            self.f[i] = self.f[i-1] + nums[i]

    def sumRange(self, i: int, j: int) -> int:
        
        return self.f[j] - self.f[i-1]
    

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)



# 198. House Robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        ln = len(nums)
        if ln == 1:
            return nums[0]
        
        f = [[0] * ln for i in range(2)]
        f[1][0]  = nums[0]
        for i in range(1,ln):
            f[1][i] = max(f[0]) + nums[i]
            f[0][i] = f[1][i-1]
        return max(max(f))
            

# 392. Is Subsequence
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if s == "":
            return True
        
        ln = len(s)
        f = [0]*ln
        f[0] = t.find(s[0])
        if f[0] == -1:
            return False
        for i in range(1,ln):
            tp = t[f[i-1]+1:].find(s[i])
            if tp == -1:
                return False
            f[i] = tp + f[i-1] + 1
        return True


# 746. Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)
        ln = len(cost)
        f = [0] * ln
        f[0] = cost[0]
        f[1] = cost[1]
        for i in range(2,ln):
            f[i] = min(f[i-1],f[i-2]) + cost[i]
        return f[-1]


# 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        f = [1] * n
        if n < 2:
            return 1
        f[1] = 2
        for i in range(2,n):
            f[i] = f[i-1] + f[i-2]
        return f[-1]


# 121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        
        p = prices[1] - prices[0]
        b = prices[0]
        for i in range(1,len(prices)):
            p = max(prices[i] - b, p)
            b = min(prices[i],b)
        return max(p,0)


# 1025. Divisor Game
class Solution:
    def divisorGame(self, N: int) -> bool:
        f = [False] * (N+1)
        for i in range(2, N+1):
            for x in range(i,0,-1):
                if i % x == 0:
                    f[i] = not f[i-x]
        return f[N]


# 53. Maximum Subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        lm = nums[0]
        gm = nums[0]
        for i in range(1,len(nums)):
            lm = max(nums[i],lm + nums[i])
            gm = max(lm,gm)
        return gm


# 53. Maximum Subarray  # this is a algrith
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ln = len(nums)
        f = [nums[0]] * ln
        for i in range(1,ln):
            f[i] = max(nums[i],f[i-1]+nums[i])
            print(f[i])
        return max(f)


# 38. Count and Say
class Solution:
    def countAndSay(self, n: int) -> str:
        def recu(s:str):
            r = ""
            ct = 0
            t = s[0]
            for i in s:
                if i == t:
                    ct += 1
                else:
                    r += str(ct) + str(t)
                    t = i
                    ct = 1
            r += str(ct) + str(t)
            return r
    
        t = '1'
        for i in range(n-1):
            t = recu(t)
        return t


# 35. Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if target <= nums[i]:
                return i
        return len(nums)


# 28. Implement strStr()
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "" :
            return 0
        if len(needle) > len(haystack):
            return -1
        
        ln = len(needle)
        lh = len(haystack)
        for i in range(lh - ln + 1):
            if haystack[i:i+ln] == needle:
                return i
        return -1


# 27. Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        while(val in nums):
            del nums[nums.index(val)]  # or nums.remove(val)
        return len(nums)


# 26. Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if (nums == []):
            return 0
        
        ct = 1
        t = nums[0]
        for i in nums:
            if(i != t):
                nums[ct] = i
                ct += 1
                t = i
        return ct



# 21. Merge Two Sorted Lists //singly-linked list.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        r = ListNode(-1)
        p = r
        while(l1 and l2):
            if(l1.val < l2.val):
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        if(l1 == None):
            p.next = l2
        else:
            p.next = l1
            
        return r.next


# 20. Valid Parentheses    ///stack
class Solution:
    def isValid(self, s: str) -> bool:
        if(s==""):
            return True
        
        p = {
            ')':'(',
            '}':'{',
            ']':'['
        }
        stack = []
        for i in s:
            if i in p.values():
                stack.append(i)
            else:
                if(stack==[] or stack.pop() != p[i]):
                    return False
        
        if(stack==[]):
            return True
                    
                        


# 14. Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if (len(strs) == 0):
            return ''
        l = len(strs[0])
        for str in strs:
            if (len(str) < l):
                l = len(str)
        
        counter = 0
        for i in range(l):
            s = strs[0][i]
            for str in strs:
                if(s!=str[i]):
                    return strs[0][0:counter]
            counter+=1
            
        return strs[0][0:counter]
                


# 9. Palindrome Number
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if(x < 0):
            return False
        init = x
        a = 0
        while(x != 0):
            a *= 10
            a += x % 10
            x = x // 10
        print(a)
        if(a == init):
            return True
        else:
            return False


# 13. Roman to Integer
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000
        }
        a = 0
        for i in range(0,len(s)-1):
            if(dic[s[i]] < dic[s[i+1]]):
                a -= dic[s[i]]
            else:
                a += dic[s[i]]
            
        return a + dic[s[len(s)-1]]
                

# 7. Reverse Integer
class Solution:
    def reverse(self, x: int) -> int:
        
        sign = 1
        
        if(x < 0):
            x = -x
            sign = -1
            
        a = 0
        while(x != 0):
            a *= 10
            a += x % 10
            x = x // 10
        x = a * sign
        
        if(x < -2**31 or x > 2**31 - 1):
            return 0 
        else:
            return x
        

#1. Two Sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ret = []
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                if(nums[i] + nums[j] == target):
                    ret.append(i)
                    ret.append(j)
        return ret
        